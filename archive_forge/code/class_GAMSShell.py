from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
@SolverFactory.register('_gams_shell', doc='Shell interface to the GAMS modeling language')
class GAMSShell(_GAMSSolver):
    """A generic shell interface to GAMS solvers."""

    def available(self, exception_flag=True):
        """True if the solver is available."""
        exe = pyomo.common.Executable('gams')
        if not exe.available():
            if not exception_flag:
                return False
            raise NameError("No 'gams' command found on system PATH - GAMS shell solver functionality is not available.")
        avail = self._run_simple_model(1)
        if not avail and exception_flag:
            raise NameError("'gams' command failed to solve a simple model - GAMS shell solver functionality is not available.")
        return avail

    def license_is_valid(self):
        return self._run_simple_model(5001)

    def _run_simple_model(self, n):
        solver_exec = self.executable()
        if solver_exec is None:
            return False
        tmpdir = mkdtemp()
        try:
            test = os.path.join(tmpdir, 'test.gms')
            with open(test, 'w') as FILE:
                FILE.write(self._simple_model(n))
            result = subprocess.run([solver_exec, test, 'curdir=' + tmpdir, 'lo=0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return not result.returncode
        finally:
            shutil.rmtree(tmpdir)
        return False

    def _default_executable(self):
        executable = pyomo.common.Executable('gams')
        if not executable:
            logger.warning("Could not locate the 'gams' executable, which is required for solver gams")
            self.enable = False
            return None
        return executable.path()

    def executable(self):
        """Returns the executable used by this solver."""
        return self._default_executable()

    def _get_version(self):
        """Returns a tuple describing the solver executable version."""
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        else:
            cmd = [solver_exec, 'audit', 'lo=3']
            results = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            return _extract_version(results.stdout)

    @staticmethod
    def _parse_special_values(value):
        if value == 1e+300 or value == 2e+300:
            return float('nan')
        if value == 3e+300:
            return float('inf')
        if value == 4e+300:
            return -float('inf')
        if value == 5e+300:
            return sys.float_info.epsilon
        return value

    def solve(self, *args, **kwds):
        """
        Solve a model via the GAMS executable.

        Keyword Arguments
        -----------------
        tee=False: bool
            Output GAMS log to stdout.
        logfile=None: str
            Filename to output GAMS log to a file.
        load_solutions=True: bool
            Load solution into model. If False, the results
            object will contain the solution data.
        keepfiles=False: bool
            Keep temporary files.
        tmpdir=None: str
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
            By default uses the system default temporary path.
        report_timing=False: bool
            Print timing reports for presolve, solver, postsolve, etc.
        io_options: dict
            Options that get passed to the writer.
            See writer in pyomo.repn.plugins.gams_writer for details.
            Updated with any other keywords passed to solve method.
            Note: put_results is not available for modification on
            GAMSShell solver.
        """
        self.available()
        if len(args) != 1:
            raise ValueError('Exactly one model must be passed to solve method of GAMSSolver.')
        model = args[0]
        options = dict()
        options.update(self.options)
        options.update(kwds)
        load_solutions = options.pop('load_solutions', True)
        tee = options.pop('tee', False)
        logfile = options.pop('logfile', None)
        keepfiles = options.pop('keepfiles', False)
        tmpdir = options.pop('tmpdir', None)
        report_timing = options.pop('report_timing', False)
        io_options = options.pop('io_options', {})
        io_options.update(options)
        initial_time = time.time()
        if logfile is not None:
            logfile = os.path.abspath(logfile)
        newdir = False
        if tmpdir is None:
            tmpdir = mkdtemp()
            newdir = True
        elif not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
            newdir = True
        output = 'model.gms'
        output_filename = os.path.join(tmpdir, output)
        lst = 'output.lst'
        lst_filename = os.path.join(tmpdir, lst)
        put_results = 'results'
        io_options['put_results'] = put_results
        io_options.setdefault('put_results_format', 'gdx' if gdxcc_available else 'dat')
        if io_options['put_results_format'] == 'gdx':
            results_filename = os.path.join(tmpdir, 'GAMS_MODEL_p.gdx')
            statresults_filename = os.path.join(tmpdir, '%s_s.gdx' % (put_results,))
        else:
            results_filename = os.path.join(tmpdir, '%s.dat' % (put_results,))
            statresults_filename = os.path.join(tmpdir, '%sstat.dat' % (put_results,))
        if isinstance(model, IBlock):
            smap_id = model.write(filename=output_filename, format=ProblemFormat.gams, _called_by_solver=True, **io_options)
            symbolMap = getattr(model, '._symbol_maps')[smap_id]
        else:
            _, smap_id = model.write(filename=output_filename, format=ProblemFormat.gams, io_options=io_options)
            symbolMap = model.solutions.symbol_map[smap_id]
        presolve_completion_time = time.time()
        if report_timing:
            print('      %6.2f seconds required for presolve' % (presolve_completion_time - initial_time))
        exe = self.executable()
        command = [exe, output, 'o=' + lst, 'curdir=' + tmpdir]
        if tee and (not logfile):
            command.append('lo=3')
        elif not tee and (not logfile):
            command.append('lo=0')
        elif not tee and logfile:
            command.append('lo=2')
        elif tee and logfile:
            command.append('lo=4')
        if logfile:
            command.append('lf=' + str(logfile))
        try:
            ostreams = [StringIO()]
            if tee:
                ostreams.append(sys.stdout)
            with TeeStream(*ostreams) as t:
                result = subprocess.run(command, stdout=t.STDOUT, stderr=t.STDERR)
            rc = result.returncode
            txt = ostreams[0].getvalue()
            if keepfiles:
                print('\nGAMS WORKING DIRECTORY: %s\n' % tmpdir)
            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    check_expr_evaluation(model, symbolMap, 'shell')
                logger.error('GAMS encountered an error during solve. Check listing file for details.')
                logger.error(txt)
                if os.path.exists(lst_filename):
                    with open(lst_filename, 'r') as FILE:
                        logger.error('GAMS Listing file:\n\n%s' % (FILE.read(),))
                raise RuntimeError('GAMS encountered an error during solve. Check listing file for details.')
            if io_options['put_results_format'] == 'gdx':
                model_soln, stat_vars = self._parse_gdx_results(results_filename, statresults_filename)
            else:
                model_soln, stat_vars = self._parse_dat_results(results_filename, statresults_filename)
        finally:
            if not keepfiles:
                if newdir:
                    shutil.rmtree(tmpdir)
                else:
                    os.remove(output_filename)
                    os.remove(lst_filename)
                    os.remove(results_filename)
                    os.remove(statresults_filename)
        solve_completion_time = time.time()
        if report_timing:
            print('      %6.2f seconds required for solver' % (solve_completion_time - presolve_completion_time))
        if isinstance(model, IBlock):
            model_suffixes = list((comp.storage_key for comp in pyomo.core.kernel.suffix.import_suffix_generator(model, active=True, descend_into=False)))
        else:
            model_suffixes = list((name for name, comp in pyomo.core.base.suffix.active_import_suffix_generator(model)))
        extract_dual = 'dual' in model_suffixes
        extract_rc = 'rc' in model_suffixes
        results = SolverResults()
        results.problem.name = output_filename
        results.problem.lower_bound = stat_vars['OBJEST']
        results.problem.upper_bound = stat_vars['OBJEST']
        results.problem.number_of_variables = stat_vars['NUMVAR']
        results.problem.number_of_constraints = stat_vars['NUMEQU']
        results.problem.number_of_nonzeros = stat_vars['NUMNZ']
        results.problem.number_of_binary_variables = None
        results.problem.number_of_integer_variables = stat_vars['NUMDVAR']
        results.problem.number_of_continuous_variables = stat_vars['NUMVAR'] - stat_vars['NUMDVAR']
        results.problem.number_of_objectives = 1
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'Only one objective is allowed.'
        obj = obj[0]
        objctvval = stat_vars['OBJVAL']
        if obj.is_minimizing():
            results.problem.sense = ProblemSense.minimize
            results.problem.upper_bound = objctvval
        else:
            results.problem.sense = ProblemSense.maximize
            results.problem.lower_bound = objctvval
        results.solver.name = 'GAMS ' + str(self.version())
        results.solver.termination_condition = None
        results.solver.message = None
        solvestat = stat_vars['SOLVESTAT']
        if solvestat == 1:
            results.solver.status = SolverStatus.ok
        elif solvestat == 2:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif solvestat == 3:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif solvestat == 5:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif solvestat == 7:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.licensingProblems
        elif solvestat == 8:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
        elif solvestat == 10:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.solverFailure
        elif solvestat == 11:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.internalSolverError
        elif solvestat == 4:
            results.solver.status = SolverStatus.warning
            results.solver.message = 'Solver quit with a problem (see LST file)'
        elif solvestat in (9, 12, 13):
            results.solver.status = SolverStatus.error
        elif solvestat == 6:
            results.solver.status = SolverStatus.unknown
        results.solver.return_code = rc
        results.solver.user_time = stat_vars['ETSOLVE']
        results.solver.system_time = None
        results.solver.wallclock_time = None
        results.solver.termination_message = None
        soln = Solution()
        modelstat = stat_vars['MODELSTAT']
        if modelstat == 1:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 2:
            results.solver.termination_condition = TerminationCondition.locallyOptimal
            soln.status = SolutionStatus.locallyOptimal
        elif modelstat in [3, 18]:
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif modelstat in [4, 5, 6, 10, 19]:
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif modelstat == 7:
            results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        elif modelstat == 8:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 9:
            results.solver.termination_condition = TerminationCondition.intermediateNonInteger
            soln.status = SolutionStatus.other
        elif modelstat == 11:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.licensingProblems
            soln.status = SolutionStatus.error
        elif modelstat in [12, 13]:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif modelstat == 14:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.noSolution
            soln.status = SolutionStatus.unknown
        elif modelstat in [15, 16, 17]:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.unsure
        else:
            soln.status = SolutionStatus.error
        soln.gap = abs(results.problem.upper_bound - results.problem.lower_bound)
        has_rc_info = True
        for sym, obj in symbolMap.bySymbol.items():
            if isinstance(model, IBlock):
                if obj.ctype is IObjective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.ctype is not IVariable:
                    continue
            else:
                if obj.parent_component().ctype is Objective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.parent_component().ctype is not Var:
                    continue
            try:
                rec = model_soln[sym]
            except KeyError:
                rec = (float('nan'), float('nan'))
            soln.variable[sym] = {'Value': float(rec[0])}
            if extract_rc and has_rc_info:
                try:
                    soln.variable[sym]['rc'] = float(rec[1])
                except ValueError:
                    has_rc_info = False
        if extract_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed() or not (c.has_lb() or c.has_ub()):
                    continue
                sym = symbolMap.getSymbol(c)
                if c.equality:
                    try:
                        rec = model_soln[sym]
                    except KeyError:
                        rec = (float('nan'), float('nan'))
                    try:
                        soln.constraint[sym] = {'dual': float(rec[1])}
                    except ValueError:
                        break
                else:
                    marg = 0
                    if c.lower is not None:
                        try:
                            rec_lo = model_soln[sym + '_lo']
                        except KeyError:
                            rec_lo = (float('nan'), float('nan'))
                        try:
                            marg -= float(rec_lo[1])
                        except ValueError:
                            marg = float('nan')
                    if c.upper is not None:
                        try:
                            rec_hi = model_soln[sym + '_hi']
                        except KeyError:
                            rec_hi = (float('nan'), float('nan'))
                        try:
                            marg += float(rec_hi[1])
                        except ValueError:
                            marg = float('nan')
                    if not math.isnan(marg):
                        soln.constraint[sym] = {'dual': marg}
                    else:
                        break
        results.solution.insert(soln)
        results._smap_id = smap_id
        results._smap = None
        if isinstance(model, IBlock):
            if len(results.solution) == 1:
                results.solution(0).symbol_map = getattr(model, '._symbol_maps')[results._smap_id]
                results.solution(0).default_variable_value = self._default_variable_value
                if load_solutions:
                    model.load_solution(results.solution(0))
            else:
                assert len(results.solution) == 0
            assert len(getattr(model, '._symbol_maps')) == 1
            delattr(model, '._symbol_maps')
            del results._smap_id
            if load_solutions and len(results.solution) == 0:
                logger.error('No solution is available')
        elif load_solutions:
            model.solutions.load_from(results)
            results._smap_id = None
            results.solution.clear()
        else:
            results._smap = model.solutions.symbol_map[smap_id]
            model.solutions.delete_symbol_map(smap_id)
        postsolve_completion_time = time.time()
        if report_timing:
            print('      %6.2f seconds required for postsolve' % (postsolve_completion_time - solve_completion_time))
            print('      %6.2f seconds required total' % (postsolve_completion_time - initial_time))
        return results

    def _parse_gdx_results(self, results_filename, statresults_filename):
        model_soln = dict()
        stat_vars = dict.fromkeys(['MODELSTAT', 'SOLVESTAT', 'OBJEST', 'OBJVAL', 'NUMVAR', 'NUMEQU', 'NUMDVAR', 'NUMNZ', 'ETSOLVE'])
        pgdx = gdxcc.new_gdxHandle_tp()
        ret = gdxcc.gdxCreateD(pgdx, os.path.dirname(self.executable()), 128)
        if not ret[0]:
            raise RuntimeError('GAMS GDX failure (gdxCreate): %s.' % ret[1])
        if os.path.exists(statresults_filename):
            ret = gdxcc.gdxOpenRead(pgdx, statresults_filename)
            if not ret[0]:
                raise RuntimeError('GAMS GDX failure (gdxOpenRead): %d.' % ret[1])
            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break
                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError('GAMS GDX failure (gdxSymbolInfo).')
                stat = ret[1]
                if not stat in stat_vars:
                    continue
                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) == 0:
                    raise RuntimeError('GAMS GDX failure (gdxDataReadRaw).')
                if stat in ('OBJEST', 'OBJVAL', 'ETSOLVE'):
                    stat_vars[stat] = self._parse_special_values(ret[2][0])
                else:
                    stat_vars[stat] = int(ret[2][0])
            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)
        if os.path.exists(results_filename):
            ret = gdxcc.gdxOpenRead(pgdx, results_filename)
            if not ret[0]:
                raise RuntimeError('GAMS GDX failure (gdxOpenRead): %d.' % ret[1])
            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break
                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) < 2:
                    raise RuntimeError('GAMS GDX failure (gdxDataReadRaw).')
                level = self._parse_special_values(ret[2][0])
                dual = self._parse_special_values(ret[2][1])
                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError('GAMS GDX failure (gdxSymbolInfo).')
                model_soln[ret[1]] = (level, dual)
            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)
        gdxcc.gdxFree(pgdx)
        return (model_soln, stat_vars)

    def _parse_dat_results(self, results_filename, statresults_filename):
        with open(statresults_filename, 'r') as statresults_file:
            statresults_text = statresults_file.read()
        stat_vars = dict()
        for line in statresults_text.splitlines()[1:]:
            items = line.split()
            try:
                stat_vars[items[0]] = float(items[1])
            except ValueError:
                stat_vars[items[0]] = float('nan')
        with open(results_filename, 'r') as results_file:
            results_text = results_file.read()
        model_soln = dict()
        for line in results_text.splitlines()[1:]:
            items = line.split()
            model_soln[items[0]] = (items[1], items[2])
        return (model_soln, stat_vars)