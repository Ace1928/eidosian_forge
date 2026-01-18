import logging
import os
import subprocess
import re
import tempfile
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
@SolverFactory.register('baron', doc='The BARON MINLP solver')
class BARONSHELL(SystemCallSolver):
    """The BARON MINLP solver"""
    _solver_info_cache = {}

    def __init__(self, **kwds):
        kwds['type'] = 'baron'
        SystemCallSolver.__init__(self, **kwds)
        self._tim_file = None
        self._valid_problem_formats = [ProblemFormat.bar]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.bar] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.bar)
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False
        self._precision_string = '.17g'

    def _get_dummy_input_files(self, check_license=False):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as fr:
                pass
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as fs:
                pass
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as ft:
                pass
            f.write('//This is a dummy .bar file created to return the baron version//\nOPTIONS {\nresults: 1;\nResName: "' + fr.name + '";\nsummary: 1;\nSumName: "' + fs.name + '";\ntimes: 1;\nTimName: "' + ft.name + '";\n}\n')
            f.write('POSITIVE_VARIABLES ')
            if check_license:
                f.write(', '.join(('x' + str(i) for i in range(11))))
            else:
                f.write('x1')
            f.write(';\n')
            f.write('OBJ: minimize x1;')
        return (f.name, fr.name, fs.name, ft.name)

    def _remove_dummy_input_files(self, fnames):
        for name in fnames:
            try:
                os.remove(name)
            except OSError:
                pass

    def license_is_valid(self):
        """Runs a check for a valid Baron license using the
        given executable (default is 'baron'). All output is
        hidden. If the test fails for any reason (including
        the executable being invalid), then this function
        will return False."""
        solver_exec = self.executable()
        if (solver_exec, 'licensed') in self._solver_info_cache:
            return self._solver_info_cache[solver_exec, 'licensed']
        if not solver_exec:
            licensed = False
        else:
            fnames = self._get_dummy_input_files(check_license=True)
            try:
                process = subprocess.Popen([solver_exec, fnames[0]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = process.communicate()
                assert stderr is None
                rc = 0
                if process.returncode:
                    rc = 1
                else:
                    stdout = stdout.decode()
                    if 'Continuing in demo mode' in stdout:
                        rc = 1
            except OSError:
                rc = 1
            finally:
                self._remove_dummy_input_files(fnames)
            licensed = not rc
        self._solver_info_cache[solver_exec, 'licensed'] = licensed
        return licensed

    def _default_executable(self):
        executable = Executable('baron')
        if not executable:
            logger.warning("Could not locate the 'baron' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if (solver_exec, 'version') in self._solver_info_cache:
            return self._solver_info_cache[solver_exec, 'version']
        if solver_exec is None:
            ver = _extract_version('')
        else:
            fnames = self._get_dummy_input_files(check_license=False)
            try:
                results = subprocess.run([solver_exec, fnames[0]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                ver = _extract_version(results.stdout)
            finally:
                self._remove_dummy_input_files(fnames)
        self._solver_info_cache[solver_exec, 'version'] = ver
        return ver

    def create_command_line(self, executable, problem_files):
        cmd = [executable, problem_files[0]]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def warm_start_capable(self):
        return False

    def _convert_problem(self, args, problem_format, valid_problem_formats, **kwds):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.baron.log')
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix='.baron.soln')
        self._tim_file = TempfileManager.create_tempfile(suffix='.baron.tim')
        solver_options = {}
        solver_options['ResName'] = self._soln_file
        solver_options['TimName'] = self._tim_file
        for key in self.options:
            lower_key = key.lower()
            if lower_key == 'resname':
                logger.warning('Ignoring user-specified option "%s=%s".  This option is set to %s, and can be overridden using the "solnfile" argument to the solve() method.' % (key, self.options[key], self._soln_file))
            elif lower_key == 'timname':
                logger.warning('Ignoring user-specified option "%s=%s".  This option is set to %s.' % (key, self.options[key], self._tim_file))
            else:
                solver_options[key] = self.options[key]
        for suffix in self._suffixes:
            if re.match(suffix, 'dual') or re.match(suffix, 'rc'):
                solver_options['WantDual'] = 1
                break
        if 'solver_options' in kwds:
            raise ValueError("Baron solver options should be set using the options object on this solver plugin. The solver_options I/O options dict for the Baron writer will be populated by this plugin's options object")
        kwds['solver_options'] = solver_options
        return OptSolver._convert_problem(self, args, problem_format, valid_problem_formats, **kwds)

    def process_logfile(self):
        results = SolverResults()
        cuts = ['Bilinear', 'LD-Envelopes', 'Multilinears', 'Convexity', 'Integrality']
        with open(self._log_file) as OUTPUT:
            for line in OUTPUT:
                for field in cuts:
                    if field in line:
                        try:
                            results.solver.statistics[field + '_cuts'] = int(line.split()[1])
                        except:
                            pass
        return results

    def process_soln_file(self, results):
        if not os.path.exists(self._soln_file):
            logger.warning('Solution file does not exist: %s' % self._soln_file)
            return
        if not os.path.exists(self._tim_file):
            logger.warning('Time file does not exist: %s' % self._tim_file)
            return
        with open(self._tim_file, 'r') as TimFile:
            with open(self._soln_file, 'r') as INPUT:
                self._process_soln_file(results, TimFile, INPUT)

    def _process_soln_file(self, results, TimFile, INPUT):
        extract_marginals = False
        extract_price = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, 'rc'):
                extract_marginals = True
                flag = True
            if re.match(suffix, 'dual'):
                extract_price = True
                flag = True
            if not flag:
                raise RuntimeError('***The BARON solver plugin cannotextract solution suffix=' + suffix)
        soln = Solution()
        line = TimFile.readline().split()
        try:
            results.problem.name = line[0]
            results.problem.number_of_constraints = int(line[1])
            results.problem.number_of_variables = int(line[2])
            try:
                results.problem.lower_bound = float(line[5])
            except ValueError:
                results.problem.lower_bound = float('-inf')
            try:
                results.problem.upper_bound = float(line[6])
            except ValueError:
                results.problem.upper_bound = float('inf')
            results.problem.missing_bounds = line[9]
            results.problem.iterations = line[10]
            results.problem.node_opt = line[11]
            results.problem.node_memmax = line[12]
            results.problem.cpu_time = float(line[13])
            results.problem.wall_time = float(line[14])
        except IndexError:
            pass
        soln.gap = results.problem.upper_bound - results.problem.lower_bound
        solver_status = line[7]
        model_status = line[8]
        objective = None
        objective_label = '__default_objective__'
        soln.objective[objective_label] = {'Value': None}
        results.problem.number_of_objectives = 1
        if objective is not None:
            results.problem.sense = 'minimizing' if objective.is_minimizing() else 'maximizing'
        if solver_status == '1':
            results.solver.status = SolverStatus.ok
        elif solver_status == '2':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
            results.solver.termination_message = 'Insufficient memory to store the number of nodes required for this search tree. Increase physical memory or change algorithmic options'
        elif solver_status == '3':
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif solver_status == '4':
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif solver_status == '5':
            results.solver.status = SolverStatus.warning
            results.solver.termination_condition = TerminationCondition.other
        elif solver_status == '6':
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
        elif solver_status == '7':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
        elif solver_status == '8':
            results.solver.status = SolverStatus.unknown
            results.solver.termination_condition = TerminationCondition.unknown
        elif solver_status == '9':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.solverFailure
        elif solver_status == '10':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
        elif solver_status == '11':
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.licensingProblems
            results.solver.termination_message = 'Run terminated because of a licensing error.'
        if model_status == '1':
            soln.status = SolutionStatus.optimal
            results.solver.termination_condition = TerminationCondition.optimal
        elif model_status == '2':
            soln.status = SolutionStatus.infeasible
            results.solver.termination_condition = TerminationCondition.infeasible
        elif model_status == '3':
            soln.status = SolutionStatus.unbounded
            results.solver.termination_condition = TerminationCondition.unbounded
        elif model_status == '4':
            soln.status = SolutionStatus.feasible
        elif model_status == '5':
            soln.status = SolutionStatus.unknown
        if results.solver.status not in [SolverStatus.error, SolverStatus.aborted]:
            var_value = []
            var_name = []
            var_marginal = []
            con_price = []
            SolvedDuringPreprocessing = False
            line = '\n'
            while line and '***' not in line:
                line = INPUT.readline()
                if 'Problem solved during preprocessing' in line:
                    SolvedDuringPreprocessing = True
            INPUT.readline()
            INPUT.readline()
            try:
                objective_value = float(INPUT.readline().split()[4])
            except IndexError:
                if solver_status == '1' and model_status in ('1', '4'):
                    logger.error("Failed to process BARON solution file: could not extract the final\nobjective value, but BARON completed normally.  This is indicative of a\nbug in Pyomo's BARON solution parser.  Please report this (along with\nthe Pyomo model and BARON version) to the Pyomo Developers.")
                return
            INPUT.readline()
            INPUT.readline()
            line = INPUT.readline()
            while line.strip() != '':
                var_value.append(float(line.split()[2]))
                line = INPUT.readline()
            has_dual_info = False
            if 'Corresponding dual solution vector is' in INPUT.readline():
                has_dual_info = True
                INPUT.readline()
                line = INPUT.readline()
                while 'Price' not in line and line.strip() != '':
                    var_marginal.append(float(line.split()[2]))
                    line = INPUT.readline()
                if 'Price' in line:
                    line = INPUT.readline()
                    line = INPUT.readline()
                    while line.strip() != '':
                        con_price.append(float(line.split()[2]))
                        line = INPUT.readline()
            while 'The best solution found is' not in INPUT.readline():
                pass
            INPUT.readline()
            INPUT.readline()
            line = INPUT.readline()
            while line.strip() != '':
                var_name.append(line.split()[0])
                line = INPUT.readline()
            assert len(var_name) == len(var_value)
            soln_variable = soln.variable
            for i, (label, val) in enumerate(zip(var_name, var_value)):
                soln_variable[label] = {'Value': val}
                if extract_marginals and has_dual_info:
                    soln_variable[label]['rc'] = var_marginal[i]
            if extract_price and has_dual_info:
                soln_constraint = soln.constraint
                for i, price_val in enumerate(con_price, 1):
                    con_label = '.c' + str(i)
                    soln_constraint[con_label] = {'dual': price_val}
            if not (SolvedDuringPreprocessing and soln.status == SolutionStatus.infeasible):
                soln.objective[objective_label] = {'Value': objective_value}
            results.solution.insert(soln)