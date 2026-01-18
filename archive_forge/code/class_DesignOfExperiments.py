from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
class DesignOfExperiments:

    def __init__(self, param_init, design_vars, measurement_vars, create_model, solver=None, prior_FIM=None, discretize_model=None, args=None):
        """
        This package enables model-based design of experiments analysis with Pyomo.
        Both direct optimization and enumeration modes are supported.
        NLP sensitivity tools, e.g.,  sipopt and k_aug, are supported to accelerate analysis via enumeration.
        It can be applied to dynamic models, where design variables are controlled throughout the experiment.

        Parameters
        ----------
        param_init:
            A  ``dictionary`` of parameter names and values.
            If they defined as indexed Pyomo variable, put the variable name and index, such as 'theta["A1"]'.
        design_vars:
            A ``DesignVariables`` which contains the Pyomo variable names and their corresponding indices
            and bounds for experiment degrees of freedom
        measurement_vars:
            A ``MeasurementVariables`` which contains the Pyomo variable names and their corresponding indices and
            bounds for experimental measurements
        create_model:
            A Python ``function`` that returns a Concrete Pyomo model, similar to the interface for ``parmest``
        solver:
            A ``solver`` object that User specified, default=None.
            If not specified, default solver is IPOPT MA57.
        prior_FIM:
            A 2D numpy array containing Fisher information matrix (FIM) for prior experiments.
            The default None means there is no prior information.
        discretize_model:
            A user-specified ``function`` that discretizes the model. Only use with Pyomo.DAE, default=None
        args:
            Additional arguments for the create_model function.
        """
        self.param = param_init
        self.design_name = design_vars.variable_names
        self.design_vars = design_vars
        self.create_model = create_model
        self.args = args
        self.measurement_vars = measurement_vars
        self.measure_name = self.measurement_vars.variable_names
        if solver:
            self.solver = solver
        else:
            self.solver = self._get_default_ipopt_solver()
        self.discretize_model = discretize_model
        if prior_FIM is None:
            self.prior_FIM = np.zeros((len(self.param), len(self.param)))
        else:
            self.prior_FIM = prior_FIM
        self._check_inputs()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

    def _check_inputs(self):
        """
        Check if the prior FIM is N*N matrix, where N is the number of parameter
        """
        if type(self.prior_FIM) != type(None):
            if np.shape(self.prior_FIM)[0] != np.shape(self.prior_FIM)[1]:
                raise ValueError('Found wrong prior information matrix shape.')
            elif np.shape(self.prior_FIM)[0] != len(self.param):
                raise ValueError('Found wrong prior information matrix shape.')

    def stochastic_program(self, if_optimize=True, objective_option='det', scale_nominal_param_value=False, scale_constant_value=1, optimize_opt=None, if_Cholesky=False, L_LB=1e-07, L_initial=None, jac_initial=None, fim_initial=None, formula='central', step=0.001, tee_opt=True):
        """
        Optimize DOE problem with design variables being the decisions.
        The DOE model is formed invasively and all scenarios are computed simultaneously.
        The function will first run a square problem with design variable being fixed at
        the given initial points (Objective function being 0), then a square problem with
        design variables being fixed at the given initial points (Objective function being Design optimality),
        and then unfix the design variable and do the optimization.

        Parameters
        ----------
        if_optimize:
            if true, continue to do optimization. else, just run square problem with given design variable values
        objective_option:
            choose from the ObjectiveLib enum,
            "det": maximizing the determinant with ObjectiveLib.det,
            "trace": or the trace of the FIM with ObjectiveLib.trace
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        optimize_opt:
            A dictionary, keys are design variables, values are True or False deciding if this design variable will be optimized as DOF or not
        if_Cholesky:
            if True, Cholesky decomposition is used for Objective function for D-optimality.
        L_LB:
            L is the Cholesky decomposition matrix for FIM, i.e. FIM = L*L.T.
            L_LB is the lower bound for every element in L.
            if FIM is positive definite, the diagonal element should be positive, so we can set a LB like 1E-10
        L_initial:
            initialize the L
        jac_initial:
            a matrix used to initialize jacobian matrix
        fim_initial:
            a matrix used to initialize FIM matrix
        formula:
            choose from "central", "forward", "backward",
            which refers to the Enum FiniteDifferenceStep.central, .forward, or .backward
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001
        tee_opt:
            if True, IPOPT console output is printed

        Returns
        -------
        analysis_square: result summary of the square problem solved at the initial point
        analysis_optimize: result summary of the optimization problem solved

        """
        self.design_values = self.design_vars.variable_names_value
        self.optimize = if_optimize
        self.objective_option = ObjectiveLib(objective_option)
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.Cholesky_option = if_Cholesky
        self.L_LB = L_LB
        self.L_initial = L_initial
        self.jac_initial = jac_initial
        self.fim_initial = fim_initial
        self.formula = FiniteDifferenceStep(formula)
        self.step = step
        self.tee_opt = tee_opt
        self.fim_scale_constant_value = self.scale_constant_value ** 2
        sp_timer = TicTocTimer()
        sp_timer.tic(msg=None)
        m = self._create_doe_model(no_obj=True)
        m, analysis_square = self._compute_stochastic_program(m, optimize_opt)
        if self.optimize:
            analysis_optimize = self._optimize_stochastic_program(m)
            dT = sp_timer.toc(msg=None)
            self.logger.info('elapsed time: %0.1f' % dT)
            return (analysis_square, analysis_optimize)
        else:
            dT = sp_timer.toc(msg=None)
            self.logger.info('elapsed time: %0.1f' % dT)
            return analysis_square

    def _compute_stochastic_program(self, m, optimize_option):
        """
        Solve the stochastic program problem as a square problem.
        """
        result_square = self._solve_doe(m, fix=True, opt_option=optimize_option)
        jac_square = self._extract_jac(m)
        analysis_square = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac_square, prior_FIM=self.prior_FIM, scale_constant_value=self.scale_constant_value)
        analysis_square.result_analysis(result=result_square)
        analysis_square.model = m
        self.analysis_square = analysis_square
        return (m, analysis_square)

    def _optimize_stochastic_program(self, m):
        """
        Solve the stochastic program problem as an optimization problem.
        """
        m = self._add_objective(m)
        result_doe = self._solve_doe(m, fix=False)
        jac_optimize = self._extract_jac(m)
        analysis_optimize = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac_optimize, prior_FIM=self.prior_FIM)
        analysis_optimize.result_analysis(result=result_doe)
        analysis_optimize.model = m
        return analysis_optimize

    def compute_FIM(self, mode='direct_kaug', FIM_store_name=None, specified_prior=None, tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1, store_output=None, read_output=None, extract_single_model=None, formula='central', step=0.001):
        """
        This function calculates the Fisher information matrix (FIM) using sensitivity information obtained
        from two possible modes (defined by the CalculationMode Enum):

            1.  sequential_finite: sequentially solve square problems and use finite difference approximation
            2.  direct_kaug: solve a single square problem then extract derivatives using NLP sensitivity theory

        Parameters
        ----------
        mode:
            supports CalculationMode.sequential_finite or CalculationMode.direct_kaug
        FIM_store_name:
            if storing the FIM in a .csv or .txt, give the file name here as a string.
        specified_prior:
            a 2D numpy array providing alternate prior matrix, default is no prior.
        tee_opt:
            if True, IPOPT console output is printed
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_output:
            if storing the output (value stored in Var 'output_record') as a pickle file, give the file name here as a string.
        read_output:
            if reading the output (value for Var 'output_record') as a pickle file, give the file name here as a string.
        extract_single_model:
            if True, the solved model outputs for each scenario are all recorded as a .csv file.
            The output file uses the name AB.csv, where string A is store_output input, B is the index of scenario.
            scenario index is the number of the scenario outputs which is stored.
        formula:
            choose from the Enum FiniteDifferenceStep.central, .forward, or .backward.
            This option is only used for CalculationMode.sequential_finite mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns
        -------
        FIM_analysis: result summary object of this solve
        """
        self.design_values = self.design_vars.variable_names_value
        self.scale_nominal_param_value = scale_nominal_param_value
        self.scale_constant_value = scale_constant_value
        self.formula = FiniteDifferenceStep(formula)
        self.mode = CalculationMode(mode)
        self.step = step
        self.optimize = False
        self.objective_option = ObjectiveLib.zero
        self.tee_opt = tee_opt
        self.FIM_store_name = FIM_store_name
        self.specified_prior = specified_prior
        self.fim_scale_constant_value = self.scale_constant_value ** 2
        square_timer = TicTocTimer()
        square_timer.tic(msg=None)
        if self.mode == CalculationMode.sequential_finite:
            FIM_analysis = self._sequential_finite(read_output, extract_single_model, store_output)
        elif self.mode == CalculationMode.direct_kaug:
            FIM_analysis = self._direct_kaug()
        dT = square_timer.toc(msg=None)
        self.logger.info('elapsed time: %0.1f' % dT)
        return FIM_analysis

    def _sequential_finite(self, read_output, extract_single_model, store_output):
        """Sequential_finite mode uses Pyomo Block to evaluate the sensitivity information."""
        if read_output:
            with open(read_output, 'rb') as f:
                output_record = pickle.load(f)
                f.close()
            jac = self._finite_calculation(output_record)
        else:
            mod = self._create_block()
            output_record = {}
            square_result = self._solve_doe(mod, fix=True)
            if extract_single_model:
                mod_name = store_output + '.csv'
                dataframe = extract_single_model(mod, square_result)
                dataframe.to_csv(mod_name)
            for s in range(len(self.scenario_list)):
                output_iter = []
                for r in self.measure_name:
                    cuid = pyo.ComponentUID(r)
                    try:
                        var_up = cuid.find_component_on(mod.block[s])
                    except:
                        raise ValueError(f'measurement {r} cannot be found in the model.')
                    output_iter.append(pyo.value(var_up))
                output_record[s] = output_iter
                output_record['design'] = self.design_values
                if store_output:
                    f = open(store_output, 'wb')
                    pickle.dump(output_record, f)
                    f.close()
            jac = self._finite_calculation(output_record)
            self.model = mod
            self.jac = jac
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior
        FIM_analysis = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac, prior_FIM=prior_in_use, store_FIM=self.FIM_store_name, scale_constant_value=self.scale_constant_value)
        return FIM_analysis

    def _direct_kaug(self):
        mod = self.create_model(model_option=ModelOptionLib.parmest)
        if self.discretize_model:
            mod = self.discretize_model(mod, block=False)
        mod.Obj = pyo.Objective(expr=0, sense=pyo.minimize)
        for par in self.param.keys():
            cuid = pyo.ComponentUID(par)
            var = cuid.find_component_on(mod)
            var.setlb(self.param[par])
            var.setub(self.param[par])
        var_name = list(self.param.keys())
        square_result = self._solve_doe(mod, fix=True)
        dsdp_re, col = get_dsdp(mod, list(self.param.keys()), self.param, tee=self.tee_opt)
        dsdp_array = dsdp_re.toarray().T
        self.dsdp = dsdp_array
        self.dsdp = col
        dsdp_extract = []
        measurement_index = []
        for mname in self.measure_name:
            try:
                kaug_no = col.index(mname)
                measurement_index.append(kaug_no)
                dsdp_extract.append(dsdp_array[kaug_no])
            except:
                self.logger.debug('The variable is fixed:  %s', mname)
                zero_sens = np.zeros(len(self.param))
                dsdp_extract.append(zero_sens)
        jac = {}
        for par in self.param.keys():
            jac[par] = []
        for d in range(len(dsdp_extract)):
            for p, par in enumerate(self.param.keys()):
                sensi = dsdp_extract[d][p] * self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= self.param[par]
                jac[par].append(sensi)
        if self.specified_prior is None:
            prior_in_use = self.prior_FIM
        else:
            prior_in_use = self.specified_prior
        FIM_analysis = FisherResults(list(self.param.keys()), self.measurement_vars, jacobian_info=None, all_jacobian_info=jac, prior_FIM=prior_in_use, store_FIM=self.FIM_store_name, scale_constant_value=self.scale_constant_value)
        self.jac = jac
        self.mod = mod
        return FIM_analysis

    def _create_block(self):
        """
        Create a pyomo Concrete model and add blocks with different parameter perturbation scenarios.

        Returns
        -------
        mod: Concrete Pyomo model
        """
        scena_gen = ScenarioGenerator(parameter_dict=self.param, formula=self.formula, step=self.step)
        self.scenario_data = scena_gen.ScenarioData
        self.scenario_list = self.scenario_data.scenario
        self.scenario_num = self.scenario_data.scena_num
        self.eps_abs = self.scenario_data.eps_abs
        self.scena_gen = scena_gen
        mod = pyo.ConcreteModel()
        mod.scenario = pyo.Set(initialize=self.scenario_data.scenario_indices)
        self.create_model(mod=mod, model_option=ModelOptionLib.stage1)

        def block_build(b, s):
            self.create_model(mod=b, model_option=ModelOptionLib.stage2)
            for par in self.param:
                cuid = pyo.ComponentUID(par)
                var = cuid.find_component_on(b)
                var.fix(self.scenario_data.scenario[s][par])
        mod.block = pyo.Block(mod.scenario, rule=block_build)
        if self.discretize_model:
            mod = self.discretize_model(mod)
        for name in self.design_name:

            def fix1(mod, s):
                cuid = pyo.ComponentUID(name)
                design_var_global = cuid.find_component_on(mod)
                design_var = cuid.find_component_on(mod.block[s])
                return design_var == design_var_global
            con_name = 'con' + name
            mod.add_component(con_name, pyo.Constraint(mod.scenario, expr=fix1))
        return mod

    def _finite_calculation(self, output_record):
        """
        Calculate Jacobian for sequential_finite mode

        Parameters
        ----------
        output_record: a dict of outputs, keys are scenario names, values are a list of measurements values
        scena_gen: an object generated by Scenario_creator class

        Returns
        -------
        jac: Jacobian matrix, a dictionary, keys are parameter names, values are a list of jacobian values with respect to this parameter
        """
        jac = {}
        for para in self.param.keys():
            involved_s = self.scenario_data.scena_num[para]
            s1 = involved_s[0]
            s2 = involved_s[1]
            list_jac = []
            for i in range(len(output_record[s1])):
                sensi = (output_record[s1][i] - output_record[s2][i]) / self.scenario_data.eps_abs[para] * self.scale_constant_value
                if self.scale_nominal_param_value:
                    sensi *= self.param[para]
                list_jac.append(sensi)
            jac[para] = list_jac
        return jac

    def _extract_jac(self, m):
        """
        Extract jacobian from the stochastic program

        Parameters
        ----------
        m: solved stochastic program model

        Returns
        -------
        JAC: the overall jacobian as a dictionary
        """
        jac = {}
        for p in self.param.keys():
            jac_para = []
            for res in m.measured_variables:
                jac_para.append(pyo.value(m.sensitivity_jacobian[p, res]))
            jac[p] = jac_para
        return jac

    def run_grid_search(self, design_ranges, mode='sequential_finite', tee_option=False, scale_nominal_param_value=False, scale_constant_value=1, store_name=None, read_name=None, store_optimality_as_csv=None, formula='central', step=0.001):
        """
        Enumerate through full grid search for any number of design variables;
        solve square problems sequentially to compute FIMs.
        It calculates FIM with sensitivity information from two modes:

            1. sequential_finite: Calculates a one scenario model multiple times for multiple scenarios.
               Sensitivity info estimated by finite difference
            2. direct_kaug: calculate sensitivity by k_aug with direct sensitivity

        Parameters
        ----------
        design_ranges:
            a ``dict``, keys are design variable names,
            values are a list of design variable values to go over
        mode:
            choose from CalculationMode.sequential_finite, .direct_kaug.
        tee_option:
            if solver console output is made
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_name:
            a string of file name. If not None, store results with this name.
            It is a pickle file containing all measurement information after solving the
            model with perturbations.
            Since there are multiple experiments, results are numbered with a scalar number,
            and the result for one grid is 'store_name(count).csv' (count is the number of count).
        read_name:
            a string of file name. If not None, read result files.
            It should be a pickle file previously generated by store_name option.
            Since there are multiple experiments, this string should be the common part of all files;
            Real name of the file is "read_name(count)", where count is the number of the experiment.
        store_optimality_as_csv:
            if True, the design criterion values of grid search results stored with this file name as a csv
        formula:
            choose from FiniteDifferenceStep.central, .forward, or .backward.
            This option is only used for CalculationMode.sequential_finite.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns
        -------
        figure_draw_object: a combined result object of class Grid_search_result
        """
        self.objective_option = ObjectiveLib.zero
        self.store_optimality_as_csv = store_optimality_as_csv
        self.fim_scale_constant_value = scale_constant_value ** 2
        result_combine = {}
        design_ranges_list = list(design_ranges.values())
        design_dimension_names = list(design_ranges.keys())
        count = 0
        failed_count = 0
        total_count = 1
        for rng in design_ranges_list:
            total_count *= len(rng)
        time_set = []
        search_design_set = product(*design_ranges_list)
        for design_set_iter in search_design_set:
            design_iter = self.design_vars.variable_names_value.copy()
            for i, names in enumerate(design_dimension_names):
                if type(names) is list or type(names) is tuple:
                    for n in names:
                        design_iter[n] = list(design_set_iter)[i]
                else:
                    design_iter[names] = list(design_set_iter)[i]
            self.design_vars.variable_names_value = design_iter
            iter_timer = TicTocTimer()
            self.logger.info('=======Iteration Number: %s =====', count + 1)
            self.logger.debug('Design variable values of this iteration: %s', design_iter)
            iter_timer.tic(msg=None)
            if store_name is None:
                store_output_name = None
            else:
                store_output_name = store_name + str(count)
            if read_name:
                read_input_name = read_name + str(count)
            else:
                read_input_name = None
            try:
                result_iter = self.compute_FIM(mode=mode, tee_opt=tee_option, scale_nominal_param_value=scale_nominal_param_value, scale_constant_value=scale_constant_value, store_output=store_output_name, read_output=read_input_name, formula=formula, step=step)
                count += 1
                result_iter.result_analysis()
                iter_t = iter_timer.toc(msg=None)
                time_set.append(iter_t)
                self.logger.info('This is run %s out of %s.', count, total_count)
                self.logger.info('The code has run  %s seconds.', sum(time_set))
                self.logger.info('Estimated remaining time:  %s seconds', sum(time_set) / (count + 1) * (total_count - count - 1))
                result_combine[tuple(design_set_iter)] = result_iter
            except:
                self.logger.warning(':::::::::::Warning: Cannot converge this run.::::::::::::')
                count += 1
                failed_count += 1
                self.logger.warning('failed count:', failed_count)
                result_combine[tuple(design_set_iter)] = None
        self.all_fim = result_combine
        figure_draw_object = GridSearchResult(design_ranges_list, design_dimension_names, result_combine, store_optimality_name=store_optimality_as_csv)
        self.logger.info('Overall wall clock time [s]:  %s', sum(time_set))
        return figure_draw_object

    def _create_doe_model(self, no_obj=True):
        """
        Add equations to compute sensitivities, FIM, and objective.

        Parameters
        -----------
        no_obj: if True, objective function is 0.

        Return
        -------
        model: the DOE model
        """
        model = self._create_block()
        model.regression_parameters = pyo.Set(initialize=list(self.param.keys()))
        model.measured_variables = pyo.Set(initialize=self.measure_name)

        def identity_matrix(m, i, j):
            if i == j:
                return 1
            else:
                return 0
        model.sensitivity_jacobian = pyo.Var(model.regression_parameters, model.measured_variables, initialize=0.1)
        if self.fim_initial:
            dict_fim_initialize = {}
            for i, bu in enumerate(model.regression_parameters):
                for j, un in enumerate(model.regression_parameters):
                    dict_fim_initialize[bu, un] = self.fim_initial[i][j]

        def initialize_fim(m, j, d):
            return dict_fim_initialize[j, d]
        if self.fim_initial:
            model.fim = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=initialize_fim)
        else:
            model.fim = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=identity_matrix)
        if type(self.L_initial) != type(None):
            dict_cho = {}
            for i, bu in enumerate(model.regression_parameters):
                for j, un in enumerate(model.regression_parameters):
                    dict_cho[bu, un] = self.L_initial[i][j]

        def init_cho(m, i, j):
            return dict_cho[i, j]
        if self.Cholesky_option:
            if type(self.L_initial) != type(None):
                model.L_ele = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=init_cho)
            else:
                model.L_ele = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=identity_matrix)
            for i, c in enumerate(model.regression_parameters):
                for j, d in enumerate(model.regression_parameters):
                    if i < j:
                        model.L_ele[c, d].fix(0.0)
                    if self.L_LB:
                        if c == d:
                            model.L_ele[c, d].setlb(self.L_LB)

        def jacobian_rule(m, p, n):
            """
            m: Pyomo model
            p: parameter
            n: response
            """
            cuid = pyo.ComponentUID(n)
            var_up = cuid.find_component_on(m.block[self.scenario_num[p][0]])
            var_lo = cuid.find_component_on(m.block[self.scenario_num[p][1]])
            if self.scale_nominal_param_value:
                return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.param[p] * self.scale_constant_value
            else:
                return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.scale_constant_value
        fim_initial_dict = {}
        for i, bu in enumerate(model.regression_parameters):
            for j, un in enumerate(model.regression_parameters):
                fim_initial_dict[bu, un] = self.prior_FIM[i][j]

        def read_prior(m, i, j):
            return fim_initial_dict[i, j]
        model.priorFIM = pyo.Expression(model.regression_parameters, model.regression_parameters, rule=read_prior)

        def fim_rule(m, p, q):
            """
            m: Pyomo model
            p: parameter
            q: parameter
            """
            return m.fim[p, q] == sum((1 / self.measurement_vars.variance[n] * m.sensitivity_jacobian[p, n] * m.sensitivity_jacobian[q, n] for n in model.measured_variables)) + m.priorFIM[p, q] * self.fim_scale_constant_value
        model.jacobian_constraint = pyo.Constraint(model.regression_parameters, model.measured_variables, rule=jacobian_rule)
        model.fim_constraint = pyo.Constraint(model.regression_parameters, model.regression_parameters, rule=fim_rule)
        return model

    def _add_objective(self, m):

        def cholesky_imp(m, c, d):
            """
            Calculate Cholesky L matrix using algebraic constraints
            """
            if list(self.param.keys()).index(c) >= list(self.param.keys()).index(d):
                return m.fim[c, d] == sum((m.L_ele[c, list(self.param.keys())[k]] * m.L_ele[d, list(self.param.keys())[k]] for k in range(list(self.param.keys()).index(d) + 1)))
            else:
                return pyo.Constraint.Skip

        def trace_calc(m):
            """
            Calculate FIM elements. Can scale each element with 1000 for performance
            """
            return m.trace == sum((m.fim[j, j] for j in m.regression_parameters))

        def det_general(m):
            """Calculate determinant. Can be applied to FIM of any size.
            det(A) = sum_{\\sigma \\in \\S_n} (sgn(\\sigma) * \\Prod_{i=1}^n a_{i,\\sigma_i})
            Use permutation() to get permutations, sgn() to get signature
            """
            r_list = list(range(len(m.regression_parameters)))
            object_p = permutations(r_list)
            list_p = list(object_p)
            det_perm = 0
            for i in range(len(list_p)):
                name_order = []
                x_order = list_p[i]
                for x in range(len(x_order)):
                    for y, element in enumerate(m.regression_parameters):
                        if x_order[x] == y:
                            name_order.append(element)
            det_perm = sum((self._sgn(list_p[d]) * sum((m.fim[each, name_order[b]] for b, each in enumerate(m.regression_parameters))) for d in range(len(list_p))))
            return m.det == det_perm
        if self.Cholesky_option:
            m.cholesky_cons = pyo.Constraint(m.regression_parameters, m.regression_parameters, rule=cholesky_imp)
            m.Obj = pyo.Objective(expr=2 * sum((pyo.log(m.L_ele[j, j]) for j in m.regression_parameters)), sense=pyo.maximize)
        elif self.objective_option == ObjectiveLib.det:
            m.det_rule = pyo.Constraint(rule=det_general)
            m.Obj = pyo.Objective(expr=pyo.log(m.det), sense=pyo.maximize)
        elif self.objective_option == ObjectiveLib.trace:
            m.trace_rule = pyo.Constraint(rule=trace_calc)
            m.Obj = pyo.Objective(expr=pyo.log(m.trace), sense=pyo.maximize)
        elif self.objective_option == ObjectiveLib.zero:
            m.Obj = pyo.Objective(expr=0)
        return m

    def _fix_design(self, m, design_val, fix_opt=True, optimize_option=None):
        """
        Fix design variable

        Parameters
        ----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix
        optimize: a dictionary, keys are design variable name, values are True or False, deciding if this design variable is optimized as DOF this time

        Returns
        -------
        m: model
        """
        for name in self.design_name:
            cuid = pyo.ComponentUID(name)
            var = cuid.find_component_on(m)
            if fix_opt:
                var.fix(design_val[name])
            elif optimize_option is None:
                var.unfix()
            elif optimize_option[name]:
                var.unfix()
        return m

    def _get_default_ipopt_solver(self):
        """Default solver"""
        solver = SolverFactory('ipopt')
        solver.options['linear_solver'] = 'ma57'
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['max_iter'] = 3000
        return solver

    def _solve_doe(self, m, fix=False, opt_option=None):
        """Solve DOE model.
        If it's a square problem, fix design variable and solve.
        Else, fix design variable and solve square problem firstly, then unfix them and solve the optimization problem

        Parameters
        ----------
        m:model
        fix: if true, solve two times (square first). Else, just solve the square problem
        opt_option: a dictionary, keys are design variable name, values are True or False,
            deciding if this design variable is optimized as DOF this time.
            If None, all design variables are optimized as DOF this time.

        Returns
        -------
        solver_results: solver results
        """
        mod = self._fix_design(m, self.design_values, fix_opt=fix, optimize_option=opt_option)
        solver_result = self.solver.solve(mod, tee=self.tee_opt)
        return solver_result

    def _sgn(self, p):
        """
        This is a helper function for stochastic_program function to compute the determinant formula.
        Give the signature of a permutation

        Parameters
        -----------
        p: the permutation (a list)

        Returns
        -------
        1 if the number of exchange is an even number
        -1 if the number is an odd number
        """
        if len(p) == 1:
            return 1
        trans = 0
        for i in range(0, len(p)):
            j = i + 1
            for j in range(j, len(p)):
                if p[i] > p[j]:
                    trans = trans + 1
        if trans % 2 == 0:
            return 1
        else:
            return -1