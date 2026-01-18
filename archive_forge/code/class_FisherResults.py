from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
class FisherResults:

    def __init__(self, parameter_names, measurements, jacobian_info=None, all_jacobian_info=None, prior_FIM=None, store_FIM=None, scale_constant_value=1, max_condition_number=1000000000000.0):
        """Analyze the FIM result for a single run

        Parameters
        ----------
        parameter_names:
            A ``list`` of parameter names
        measurements:
            A ``MeasurementVariables`` which contains the Pyomo variable names and their corresponding indices and
            bounds for experimental measurements
        jacobian_info:
            the jacobian for this measurement object
        all_jacobian_info:
            the overall jacobian
        prior_FIM:
            if there's prior FIM to be added
        store_FIM:
            if storing the FIM in a .csv or .txt, give the file name here as a string
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        max_condition_number:
            max condition number
        """
        self.parameter_names = parameter_names
        self.measurements = measurements
        self.measurement_variables = measurements.variable_names
        if jacobian_info is None:
            self.jaco_information = all_jacobian_info
        else:
            self.jaco_information = jacobian_info
        self.all_jacobian_info = all_jacobian_info
        self.prior_FIM = prior_FIM
        self.store_FIM = store_FIM
        self.scale_constant_value = scale_constant_value
        self.fim_scale_constant_value = scale_constant_value ** 2
        self.max_condition_number = max_condition_number
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.WARN)

    def result_analysis(self, result=None):
        """Calculate FIM from Jacobian information. This is for grid search (combined models) results

        Parameters
        ----------
        result:
            solver status returned by IPOPT
        """
        self.result = result
        self.doe_result = None
        no_param = len(self.parameter_names)
        fim = np.zeros((no_param, no_param))
        Q_all = []
        for par in self.parameter_names:
            Q_all.append(self.jaco_information[par])
        n = len(self.parameter_names)
        Q_all = np.array(list((self.jaco_information[p] for p in self.parameter_names))).T
        for i, mea_name in enumerate(self.measurement_variables):
            fim += 1 / self.measurements.variance[str(mea_name)] * (Q_all[i, :].reshape(n, 1) @ Q_all[i, :].reshape(n, 1).T)
        if self.prior_FIM is not None:
            try:
                fim = fim + self.prior_FIM
                self.logger.info('Existed information has been added.')
            except:
                raise ValueError('Check the shape of prior FIM.')
        if np.linalg.cond(fim) > self.max_condition_number:
            self.logger.info('Warning: FIM is near singular. The condition number is: %s ;', np.linalg.cond(fim))
            self.logger.info('A condition number bigger than %s is considered near singular.', self.max_condition_number)
        self._print_FIM_info(fim)
        if self.result is not None:
            self._get_solver_info()
        if self.store_FIM is not None:
            self._store_FIM()

    def subset(self, measurement_subset):
        """Create new FisherResults object corresponding to provided measurement_subset.
        This requires that measurement_subset is a true subset of the original measurement object.

        Parameters
        ----------
        measurement_subset: Instance of Measurements class

        Returns
        -------
        new_result: New instance of FisherResults
        """
        self.measurements.check_subset(measurement_subset)
        small_jac = self._split_jacobian(measurement_subset)
        FIM_subset = FisherResults(self.parameter_names, measurement_subset, jacobian_info=small_jac, prior_FIM=self.prior_FIM, store_FIM=self.store_FIM, scale_constant_value=self.scale_constant_value, max_condition_number=self.max_condition_number)
        return FIM_subset

    def _split_jacobian(self, measurement_subset):
        """
        Split jacobian

        Parameters
        ----------
        measurement_subset: the object of the measurement subsets

        Returns
        -------
        jaco_info: split Jacobian
        """
        jaco_info = {}
        for par in self.parameter_names:
            jaco_info[par] = []
            for name in measurement_subset.variable_names:
                try:
                    n_all_measure = self.measurement_variables.index(name)
                    jaco_info[par].append(self.all_jacobian_info[par][n_all_measure])
                except:
                    raise ValueError('Measurement ', name, ' is not in original measurement set.')
        return jaco_info

    def _print_FIM_info(self, FIM):
        """
        using a dictionary to store all FIM information

        Parameters
        ----------
        FIM: the Fisher Information Matrix, needs to be P.D. and symmetric

        Returns
        -------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list of FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
        """
        eig = np.linalg.eigvals(FIM)
        self.FIM = FIM
        self.trace = np.trace(FIM)
        self.det = np.linalg.det(FIM)
        self.min_eig = min(eig)
        self.cond = max(eig) / min(eig)
        self.eig_vals = eig
        self.eig_vecs = np.linalg.eig(FIM)[1]
        self.logger.info('FIM: %s; \n Trace: %s; \n Determinant: %s;', self.FIM, self.trace, self.det)
        self.logger.info('Condition number: %s; \n Min eigenvalue: %s.', self.cond, self.min_eig)

    def _solution_info(self, m, dv_set):
        """
        Solution information. Only for optimization problem

        Parameters
        ----------
        m: model
        dv_set: design variable dictionary

        Returns
        -------
        model_info: model solutions dictionary containing the following key:value pairs
            -['obj']: a scalar number of objective function value
            -['det']: a scalar number of determinant calculated by the model (different from FIM_info['det'] which
            is calculated by numpy)
            -['trace']: a scalar number of trace calculated by the model
            -[design variable name]: a list of design variable solution
        """
        self.obj_value = value(m.obj)
        if self.obj == 'det':
            self.obj_det = np.exp(value(m.obj)) / self.fim_scale_constant_value ** len(self.parameter_names)
        elif self.obj == 'trace':
            self.obj_trace = np.exp(value(m.obj)) / self.fim_scale_constant_value
        design_variable_names = list(dv_set.keys())
        dv_times = list(dv_set.values())
        solution = {}
        for d, dname in enumerate(design_variable_names):
            sol = []
            if dv_times[d] is not None:
                for t, time in enumerate(dv_times[d]):
                    newvar = getattr(m, dname)[time]
                    sol.append(value(newvar))
            else:
                newvar = getattr(m, dname)
                sol.append(value(newvar))
            solution[dname] = sol
        self.solution = solution

    def _store_FIM(self):
        store_dict = {}
        for i, name in enumerate(self.parameter_names):
            store_dict[name] = self.FIM[i]
        FIM_store = pd.DataFrame(store_dict)
        FIM_store.to_csv(self.store_FIM, index=False)

    def _get_solver_info(self):
        """
        Solver information dictionary

        Return:
        ------
        solver_status: a solver information dictionary containing the following key:value pairs
            -['square']: a string of square result solver status
            -['doe']: a string of doe result solver status
        """
        if self.result.solver.status == SolverStatus.ok and self.result.solver.termination_condition == TerminationCondition.optimal:
            self.status = 'converged'
        elif self.result.solver.termination_condition == TerminationCondition.infeasible:
            self.status = 'infeasible'
        else:
            self.status = self.result.solver.status