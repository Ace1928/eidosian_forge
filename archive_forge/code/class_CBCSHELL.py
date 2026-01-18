import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
@SolverFactory.register('_cbc_shell', doc='Shell interface to the CBC LP/MIP solver')
class CBCSHELL(SystemCallSolver):
    """Shell interface to the CBC LP/MIP solver"""

    def __init__(self, **kwds):
        kwds['type'] = 'cbc'
        super(CBCSHELL, self).__init__(**kwds)
        self._warm_start_solve = False
        self._warm_start_file_name = None
        self._valid_problem_formats = [ProblemFormat.cpxlp, ProblemFormat.nl]
        self._valid_result_formats = {ProblemFormat.cpxlp: [ResultsFormat.soln], ProblemFormat.nl: [ResultsFormat.sol]}
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = False
        self._capabilities.quadratic_constraint = False
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False
        self.set_problem_format(ProblemFormat.cpxlp)

    def set_problem_format(self, format):
        super(CBCSHELL, self).set_problem_format(format)
        if self._problem_format == ProblemFormat.cpxlp:
            self._capabilities.sos1 = False
            self._capabilities.sos2 = False
        else:
            self._capabilities.sos1 = True
            self._capabilities.sos2 = True
        if self._problem_format == ProblemFormat.nl:
            if self._compiled_with_asl():
                _ver = self.version()
                if not _ver or _ver[:3] < (2, 7, 0):
                    if _ver is None:
                        _ver_str = '<unknown>'
                    else:
                        _ver_str = '.'.join((str(i) for i in _ver))
                    logger.warning(f'found CBC version {_ver_str} < 2.7; ASL support disabled (falling back on LP interface).')
                    logger.warning('Upgrade CBC to activate ASL support in this plugin')
                    self.set_problem_format(ProblemFormat.cpxlp)
            else:
                logger.warning('CBC solver is not compiled with ASL interface (falling back on LP interface).')
                self.set_problem_format(ProblemFormat.cpxlp)

    def _default_results_format(self, prob_format):
        if prob_format == ProblemFormat.nl:
            return ResultsFormat.sol
        return ResultsFormat.soln

    def warm_start_capable(self):
        if self._problem_format != ProblemFormat.cpxlp:
            return False
        _ver = self.version()
        return _ver and _ver >= (2, 8, 0, 0)

    def _write_soln_file(self, instance, filename):
        if isinstance(instance, IBlock):
            smap = getattr(instance, '._symbol_maps')[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        column_index = 0
        with open(filename, 'w') as solnfile:
            for var in instance.component_data_objects(Var):
                if var.value and (var.is_integer() or var.is_binary()) and (id(var) in byObject):
                    name = byObject[id(var)]
                    solnfile.write('{} {} {}\n'.format(column_index, name, var.value))
                    column_index += 1

    def _warm_start(self, instance):
        self._write_soln_file(instance, self._warm_start_file_name)

    def _presolve(self, *args, **kwds):
        TempfileManager.push()
        self._warm_start_solve = kwds.pop('warmstart', False)
        self._warm_start_file_name = kwds.pop('warmstart_file', None)
        user_warmstart = False
        if self._warm_start_file_name is not None:
            user_warmstart = True
        if self._warm_start_solve and isinstance(args[0], str):
            pass
        elif self._warm_start_solve and (not isinstance(args[0], str)):
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = TempfileManager.create_tempfile(suffix='.cbc.soln')
        if self._warm_start_file_name is not None:
            _drive, _path = os.path.splitdrive(self._warm_start_file_name)
            if _drive:
                _cwd_drive = os.path.splitdrive(os.getcwd())[0]
                if _cwd_drive.lower() == _drive.lower():
                    self._warm_start_file_name = _path
                else:
                    logger.warning('warmstart_file points to a file on a drive different from the current working directory.  CBC is likely to (silently) ignore the warmstart.')
        super(CBCSHELL, self)._presolve(*args, **kwds)
        if len(args) > 0 and (not isinstance(args[0], str)):
            if len(args) != 1:
                raise ValueError('CBCplugin _presolve method can only handle a single problem instance - %s were supplied' % (len(args),))
            if self._warm_start_solve and (not user_warmstart):
                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing is True:
                    print('Warm start write time=%.2f seconds' % (end_time - start_time))

    def _default_executable(self):
        executable = Executable('cbc')
        if not executable:
            logger.warning("Could not locate the 'cbc' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        results = subprocess.run([self.executable(), '-stop'], timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        _version = _extract_version(results.stdout)
        if _version is None:
            return _extract_version('')
        return _version

    def _compiled_with_asl(self):
        results = subprocess.run([self.executable(), 'dummy', '-AMPL', '-stop'], timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return 'No match for AMPL'.lower() not in results.stdout.lower()

    def create_command_line(self, executable, problem_files):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.cbc.log')
        problem_filename_prefix = problem_files[0]
        if '.' in problem_filename_prefix:
            tmp = problem_filename_prefix.split('.')
            if len(tmp) > 2:
                problem_filename_prefix = '.'.join(tmp[:-1])
            else:
                problem_filename_prefix = tmp[0]
        if self._results_format is ResultsFormat.sol:
            self._soln_file = problem_filename_prefix + '.sol'
        else:
            self._soln_file = problem_filename_prefix + '.soln'
        if self._results_format is ResultsFormat.sol:
            self._results_file = self._soln_file

        def _check_and_escape_options(options):
            for key, val in self.options.items():
                tmp_k = str(key)
                _bad = ' ' in tmp_k
                tmp_v = str(val)
                if ' ' in tmp_v:
                    if '"' in tmp_v:
                        if "'" in tmp_v:
                            _bad = True
                        else:
                            tmp_v = "'" + tmp_v + "'"
                    else:
                        tmp_v = '"' + tmp_v + '"'
                if _bad:
                    raise ValueError('Unable to properly escape solver option:\n\t%s=%s' % (key, val))
                yield (tmp_k, tmp_v)
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        if self._problem_format == ProblemFormat.nl:
            cmd.append(problem_files[0])
            cmd.append('-AMPL')
            if self._timelimit is not None and self._timelimit > 0.0:
                cmd.extend(['-sec', str(self._timelimit)])
                cmd.extend(['-timeMode', 'elapsed'])
            if 'debug' in self.options:
                cmd.extend(['-log', '5'])
            for key, val in _check_and_escape_options(self.options):
                if key == 'solver':
                    continue
                cmd.append(key + '=' + val)
            os.environ['cbc_options'] = 'printingOptions=all'
        else:
            if self._timelimit is not None and self._timelimit > 0.0:
                cmd.extend(['-sec', str(self._timelimit)])
                cmd.extend(['-timeMode', 'elapsed'])
            if 'debug' in self.options:
                cmd.extend(['-log', '5'])
            action_options = []
            for key, val in _check_and_escape_options(self.options):
                if val.strip() != '':
                    cmd.extend(['-' + key, val])
                else:
                    action_options.append('-' + key)
            cmd.extend(['-printingOptions', 'all', '-import', problem_files[0]])
            cmd.extend(action_options)
            if self._warm_start_solve:
                cmd.extend(['-mipstart', self._warm_start_file_name])
            cmd.extend(['-stat=1', '-solve', '-solu', self._soln_file])
        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()
        if self._problem_format is ProblemFormat.nl:
            return results
        soln = Solution()
        OUTPUT = open(self._log_file)
        output = ''.join(OUTPUT.readlines())
        OUTPUT.close()
        results.problem.sense = ProblemSense.minimize
        results.problem.name = None
        optim_value = float('inf')
        lower_bound = None
        upper_bound = None
        gap = None
        nodes = None
        for line in output.split('\n'):
            tokens = tuple(re.split('[ \t]+', line.strip()))
            n_tokens = len(tokens)
            if n_tokens > 1:
                if n_tokens > 4 and tokens[:4] == ('Continuous', 'objective', 'value', 'is'):
                    lower_bound = _float(tokens[4])
                elif n_tokens > 12 and tokens[1:3] == ('Search', 'completed') and (tokens[4:6] == ('best', 'objective')) and (tokens[9] == 'iterations') and (tokens[12] == 'nodes'):
                    optim_value = _float(tokens[6][:-1])
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[8])
                    nodes = int(tokens[11])
                elif tokens[1] == 'Exiting' and n_tokens > 4:
                    if tokens[2:4] == ('on', 'maximum'):
                        results.solver.termination_condition = {'nodes': TerminationCondition.maxEvaluations, 'time': TerminationCondition.maxTimeLimit, 'solutions': TerminationCondition.other, 'iterations': TerminationCondition.maxIterations}.get(tokens[4], TerminationCondition.other)
                elif n_tokens >= 4 and tokens[1:4] == ('Integer', 'solution', 'of'):
                    optim_value = _float(tokens[4])
                    try:
                        results.solver.statistics.black_box.number_of_iterations = int(tokens[tokens.index('iterations') - 1])
                        nodes = int(tokens[tokens.index('nodes') - 1])
                    except ValueError:
                        pass
                elif n_tokens > 15 and tokens[1:3] == ('Partial', 'search') and (tokens[4:6] == ('best', 'objective')) and (tokens[7:9] == ('(best', 'possible')) and (tokens[12] == 'iterations') and (tokens[15] == 'nodes'):
                    optim_value = _float(tokens[6])
                    lower_bound = _float(tokens[9][:-2])
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[11])
                    nodes = int(tokens[14])
                elif n_tokens > 12 and tokens[1] == 'After' and (tokens[3] == 'nodes,') and (tokens[8:10] == ('best', 'solution,')) and (tokens[10:12] == ('best', 'possible')):
                    nodes = int(tokens[2])
                    optim_value = _float(tokens[7])
                    lower_bound = _float(tokens[12])
                elif tokens[0] == 'Current' and n_tokens == 10 and (tokens[1] == 'default') and (tokens[2] == '(if') and (results.problem.name is None):
                    results.problem.name = tokens[-1]
                    if '.' in results.problem.name:
                        parts = results.problem.name.split('.')
                        if len(parts) > 2:
                            results.problem.name = '.'.join(parts[:-1])
                        else:
                            results.problem.name = results.problem.name.split('.')[0]
                    if '/' in results.problem.name:
                        results.problem.name = results.problem.name.split('/')[-1]
                    if '\\' in results.problem.name:
                        results.problem.name = results.problem.name.split('\\')[-1]
                elif tokens[0] == 'Presolve':
                    if n_tokens > 9 and tokens[3] == 'rows,' and (tokens[6] == 'columns'):
                        results.problem.number_of_variables = int(tokens[4]) - int(tokens[5][1:-1])
                        results.problem.number_of_constraints = int(tokens[1]) - int(tokens[2][1:-1])
                        results.problem.number_of_objectives = 1
                    elif n_tokens > 6 and tokens[6] == 'infeasible':
                        soln.status = SolutionStatus.infeasible
                elif n_tokens > 11 and tokens[:2] == ('Problem', 'has') and (tokens[3] == 'rows,') and (tokens[5] == 'columns') and (tokens[7:9] == ('with', 'objective)')):
                    results.problem.number_of_variables = int(tokens[4])
                    results.problem.number_of_constraints = int(tokens[2])
                    results.problem.number_of_nonzeros = int(tokens[6][1:])
                    results.problem.number_of_objectives = 1
                elif n_tokens > 8 and tokens[:3] == ('Original', 'problem', 'has') and (tokens[4] == 'integers') and (tokens[6:9] == ('of', 'which', 'binary)')):
                    results.problem.number_of_integer_variables = int(tokens[3])
                    results.problem.number_of_binary_variables = int(tokens[5][1:])
                elif n_tokens == 5 and tokens[3] == 'NAME':
                    results.problem.name = tokens[4]
                elif 'CoinLpIO::readLp(): Maximization problem reformulated as minimization' in ' '.join(tokens):
                    results.problem.sense = ProblemSense.maximize
                elif n_tokens > 3 and tokens[:2] == ('Result', '-'):
                    if tokens[2:4] in [('Run', 'abandoned'), ('User', 'ctrl-c')]:
                        results.solver.termination_condition = TerminationCondition.userInterrupt
                    if n_tokens > 4:
                        if tokens[2:5] == ('Optimal', 'solution', 'found'):
                            soln.status = SolutionStatus.optimal
                        elif tokens[2:5] in [('Linear', 'relaxation', 'infeasible'), ('Problem', 'proven', 'infeasible')]:
                            soln.status = SolutionStatus.infeasible
                        elif tokens[2:5] == ('Linear', 'relaxation', 'unbounded'):
                            soln.status = SolutionStatus.unbounded
                        elif n_tokens > 5 and tokens[2:4] == ('Stopped', 'on') and (tokens[5] == 'limit'):
                            results.solver.termination_condition = {'node': TerminationCondition.maxEvaluations, 'time': TerminationCondition.maxTimeLimit, 'solution': TerminationCondition.other, 'iterations': TerminationCondition.maxIterations}.get(tokens[4], TerminationCondition.other)
                    elif n_tokens > 3 and tokens[2] == 'Finished':
                        soln.status = SolutionStatus.optimal
                        optim_value = _float(tokens[4])
                elif n_tokens >= 3 and tokens[:2] == ('Objective', 'value:'):
                    optim_value = _float(tokens[2])
                elif n_tokens >= 4 and tokens[:4] == ('No', 'feasible', 'solution', 'found'):
                    soln.status = SolutionStatus.infeasible
                elif n_tokens > 2 and tokens[:2] == ('Lower', 'bound:'):
                    if lower_bound is None:
                        results.problem.lower_bound = _float(tokens[2])
                elif tokens[0] == 'Gap:':
                    gap = _float(tokens[1])
                elif n_tokens > 2 and tokens[:2] == ('Enumerated', 'nodes:'):
                    nodes = int(tokens[2])
                elif n_tokens > 2 and tokens[:2] == ('Total', 'iterations:'):
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[2])
                elif n_tokens > 3 and tokens[:3] == ('Time', '(CPU', 'seconds):'):
                    results.solver.system_time = _float(tokens[3])
                elif n_tokens > 3 and tokens[:3] == ('Time', '(Wallclock', 'Seconds):'):
                    results.solver.wallclock_time = _float(tokens[3])
                elif n_tokens > 4 and tokens[:4] == ('Total', 'time', '(CPU', 'seconds):'):
                    results.solver.system_time = _float(tokens[4])
                    if n_tokens > 7 and tokens[5:7] == ('(Wallclock', 'seconds):'):
                        results.solver.wallclock_time = _float(tokens[7])
                elif tokens[0] == 'Optimal':
                    if n_tokens > 4 and tokens[2] == 'objective' and (tokens[4] != 'and'):
                        soln.status = SolutionStatus.optimal
                        optim_value = _float(tokens[4])
                    elif n_tokens > 5 and tokens[1] == 'objective' and (tokens[5] == 'iterations'):
                        soln.status = SolutionStatus.optimal
                        optim_value = _float(tokens[2])
                        results.solver.statistics.black_box.number_of_iterations = int(tokens[4])
                elif tokens[0] == 'sys' and n_tokens == 2:
                    results.solver.system_time = _float(tokens[1])
                elif tokens[0] == 'user' and n_tokens == 2:
                    results.solver.user_time = _float(tokens[1])
                elif n_tokens == 10 and 'Presolve' in tokens and ('iterations' in tokens) and (tokens[0] == 'Optimal') and ('objective' == tokens[1]):
                    soln.status = SolutionStatus.optimal
                    optim_value = _float(tokens[2])
                results.solver.user_time = -1.0
        if results.problem.name is None:
            results.problem.name = 'unknown'
        if soln.status is SolutionStatus.optimal:
            results.solver.termination_message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
            results.solver.termination_condition = TerminationCondition.optimal
            results.solver.status = SolverStatus.ok
            if gap is None:
                lower_bound = optim_value
                gap = 0.0
        elif soln.status == SolutionStatus.infeasible:
            results.solver.termination_message = 'Model was proven to be infeasible.'
            results.solver.termination_condition = TerminationCondition.infeasible
            results.solver.status = SolverStatus.warning
        elif soln.status == SolutionStatus.unbounded:
            results.solver.termination_message = 'Model was proven to be unbounded.'
            results.solver.termination_condition = TerminationCondition.unbounded
            results.solver.status = SolverStatus.warning
        elif results.solver.termination_condition in [TerminationCondition.maxTimeLimit, TerminationCondition.maxEvaluations, TerminationCondition.other, TerminationCondition.maxIterations]:
            results.solver.status = SolverStatus.aborted
            soln.status = SolutionStatus.stoppedByLimit
            if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                results.solver.termination_message = 'Optimization terminated because the time expended exceeded the value specified in the seconds parameter.'
            elif results.solver.termination_condition == TerminationCondition.maxEvaluations:
                results.solver.termination_message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the maxNodes parameter'
            elif results.solver.termination_condition == TerminationCondition.other:
                results.solver.termination_message = 'Optimization terminated because the number of solutions found reached the value specified in the maxSolutions parameter.'
            elif results.solver.termination_condition == TerminationCondition.maxIterations:
                results.solver.termination_message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the maxIterations parameter.'
        soln.gap = gap
        if results.problem.sense == ProblemSense.minimize:
            upper_bound = optim_value
        elif results.problem.sense == ProblemSense.maximize:
            _ver = self.version()
            if _ver and _ver[:3] < (2, 10, 2):
                optim_value *= -1
                upper_bound = None if lower_bound is None else -lower_bound
            else:
                upper_bound = None if lower_bound is None else lower_bound
            lower_bound = optim_value
        soln.objective['__default_objective__'] = {'Value': optim_value}
        results.problem.lower_bound = lower_bound
        results.problem.upper_bound = upper_bound
        results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = nodes
        results.solver.statistics.branch_and_bound.number_of_created_subproblems = nodes
        if soln.status in [SolutionStatus.optimal, SolutionStatus.stoppedByLimit, SolutionStatus.unknown, SolutionStatus.other]:
            results.solution.insert(soln)
        return results

    def process_soln_file(self, results):
        extract_duals = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, 'dual'):
                extract_duals = True
                flag = True
            if re.match(suffix, 'rc'):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError('***CBC solver plugin cannot extract solution suffix=' + suffix)
        if self._results_format is ResultsFormat.sol:
            return
        if len(results.solution) > 0:
            solution = results.solution(0)
        else:
            solution = Solution()
        results.problem.number_of_objectives = 1
        processing_constraints = None
        header_processed = False
        optim_value = None
        try:
            INPUT = open(self._soln_file, 'r')
        except IOError:
            INPUT = []
        _ver = self.version()
        invert_objective_sense = results.problem.sense == ProblemSense.maximize and (_ver and _ver[:3] < (2, 10, 2))
        for line in INPUT:
            tokens = tuple(re.split('[ \t]+', line.strip()))
            n_tokens = len(tokens)
            if not header_processed:
                if tokens[0] == 'Optimal':
                    results.solver.termination_condition = TerminationCondition.optimal
                    results.solver.status = SolverStatus.ok
                    results.solver.termination_message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
                    solution.status = SolutionStatus.optimal
                    optim_value = _float(tokens[-1])
                elif tokens[0] in ('Infeasible', 'PrimalInfeasible') or (n_tokens > 1 and tokens[0:2] == ('Integer', 'infeasible')):
                    results.solver.termination_message = 'Model was proven to be infeasible.'
                    results.solver.termination_condition = TerminationCondition.infeasible
                    results.solver.status = SolverStatus.warning
                    solution.status = SolutionStatus.infeasible
                    INPUT.close()
                    return
                elif tokens[0] == 'Unbounded' or (n_tokens > 2 and tokens[0] == 'Problem' and (tokens[2] == 'unbounded')) or (n_tokens > 1 and tokens[0:2] == ('Dual', 'infeasible')):
                    results.solver.termination_message = 'Model was proven to be unbounded.'
                    results.solver.termination_condition = TerminationCondition.unbounded
                    results.solver.status = SolverStatus.warning
                    solution.status = SolutionStatus.unbounded
                    INPUT.close()
                    return
                elif n_tokens > 2 and tokens[0:2] == ('Stopped', 'on'):
                    optim_value = _float(tokens[-1])
                    solution.gap = None
                    results.solver.status = SolverStatus.aborted
                    solution.status = SolutionStatus.stoppedByLimit
                    if tokens[2] == 'time':
                        results.solver.termination_message = 'Optimization terminated because the time expended exceeded the value specified in the seconds parameter.'
                        results.solver.termination_condition = TerminationCondition.maxTimeLimit
                    elif tokens[2] == 'iterations':
                        if results.solver.termination_condition not in [TerminationCondition.maxEvaluations, TerminationCondition.other, TerminationCondition.maxIterations]:
                            results.solver.termination_message = 'Optimization terminated because a limit was hit'
                            results.solver.termination_condition = TerminationCondition.maxIterations
                    elif tokens[2] == 'difficulties':
                        results.solver.termination_condition = TerminationCondition.solverFailure
                        results.solver.status = SolverStatus.error
                        solution.status = SolutionStatus.error
                    elif tokens[2] == 'ctrl-c':
                        results.solver.termination_message = 'Optimization was terminated by the user.'
                        results.solver.termination_condition = TerminationCondition.userInterrupt
                        solution.status = SolutionStatus.unknown
                    else:
                        results.solver.termination_condition = TerminationCondition.unknown
                        results.solver.status = SolverStatus.unknown
                        solution.status = SolutionStatus.unknown
                        results.solver.termination_message = ' '.join(tokens)
                        print('***WARNING: CBC plugin currently not processing solution status=Stopped correctly. Full status line is: {}'.format(line.strip()))
                    if n_tokens > 8 and tokens[3:9] == ('(no', 'integer', 'solution', '-', 'continuous', 'used)'):
                        results.solver.termination_message = 'Optimization terminated because a limit was hit, however it had not found an integer solution yet.'
                        results.solver.termination_condition = TerminationCondition.intermediateNonInteger
                        solution.status = SolutionStatus.other
                else:
                    results.solver.termination_condition = TerminationCondition.unknown
                    results.solver.status = SolverStatus.unknown
                    solution.status = SolutionStatus.unknown
                    results.solver.termination_message = ' '.join(tokens)
                    print('***WARNING: CBC plugin currently not processing solution status={} correctly. Full status line is: {}'.format(tokens[0], line.strip()))
            try:
                row_number = int(tokens[0])
                if row_number == 0:
                    if processing_constraints is None:
                        processing_constraints = True
                    elif processing_constraints is True:
                        processing_constraints = False
                    else:
                        raise RuntimeError('CBC plugin encountered unexpected line=(' + line.strip() + ') in solution file=' + self._soln_file + '; constraint and variable sections already processed!')
            except ValueError:
                if tokens[0] in ('Optimal', 'Infeasible', 'Unbounded', 'Stopped', 'Integer', 'Status'):
                    if optim_value is not None:
                        if invert_objective_sense:
                            optim_value *= -1
                        solution.objective['__default_objective__'] = {'Value': optim_value}
                    header_processed = True
            if processing_constraints is True and extract_duals is True:
                if n_tokens == 4:
                    pass
                elif n_tokens == 5 and tokens[0] == '**':
                    tokens = tokens[1:]
                else:
                    raise RuntimeError('Unexpected line format encountered in CBC solution file - line=' + line)
                constraint = tokens[1]
                constraint_ax = _float(tokens[2])
                constraint_dual = _float(tokens[3])
                if invert_objective_sense:
                    constraint_dual *= -1
                if constraint[:2] == 'c_':
                    solution.constraint[constraint] = {'Dual': constraint_dual}
                elif constraint[:2] == 'r_':
                    existing_constraint_dual_dict = solution.constraint.get('r_l_' + constraint[4:], None)
                    if existing_constraint_dual_dict:
                        existing_constraint_dual = existing_constraint_dual_dict['Dual']
                        if abs(constraint_dual) > abs(existing_constraint_dual):
                            solution.constraint['r_l_' + constraint[4:]] = {'Dual': constraint_dual}
                    else:
                        solution.constraint['r_l_' + constraint[4:]] = {'Dual': constraint_dual}
            elif processing_constraints is False:
                if n_tokens == 4:
                    pass
                elif n_tokens == 5 and tokens[0] == '**':
                    tokens = tokens[1:]
                else:
                    raise RuntimeError('Unexpected line format encountered in CBC solution file - line=' + line)
                variable_name = tokens[1]
                variable_value = _float(tokens[2])
                variable = solution.variable[variable_name] = {'Value': variable_value}
                if extract_reduced_costs is True:
                    variable_reduced_cost = _float(tokens[3])
                    if invert_objective_sense:
                        variable_reduced_cost *= -1
                    variable['Rc'] = variable_reduced_cost
            elif header_processed is True:
                pass
            else:
                raise RuntimeError('CBC plugin encountered unexpected line=(' + line.strip() + ') in solution file=' + self._soln_file + '; expecting header, but found data!')
        if not type(INPUT) is list:
            INPUT.close()
        if len(results.solution) == 0 and solution.status in [SolutionStatus.optimal, SolutionStatus.stoppedByLimit, SolutionStatus.unknown, SolutionStatus.other]:
            results.solution.insert(solution)

    def _postsolve(self):
        results = super(CBCSHELL, self)._postsolve()
        TempfileManager.pop(remove=not self._keepfiles)
        return results