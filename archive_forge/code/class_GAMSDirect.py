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
@SolverFactory.register('_gams_direct', doc='Direct python interface to the GAMS modeling language')
class GAMSDirect(_GAMSSolver):
    """
    A generic python interface to GAMS solvers.

    Visit Python API page on gams.com for installation help.
    """

    def available(self, exception_flag=True):
        """True if the solver is available."""
        try:
            from gams import GamsWorkspace, DebugLevel
        except ImportError as e:
            if not exception_flag:
                return False
            raise ImportError('Import of gams failed - GAMS direct solver functionality is not available.\nGAMS message: %s' % (e,))
        avail = self._run_simple_model(1)
        if not avail and exception_flag:
            raise NameError("'gams' command failed to solve a simple model - GAMS shell solver functionality is not available.")
        return avail

    def license_is_valid(self):
        return self._run_simple_model(5001)

    def _get_version(self):
        """Returns a tuple describing the solver executable version."""
        if not self.available(exception_flag=False):
            return _extract_version('')
        from gams import GamsWorkspace
        ws = GamsWorkspace()
        version = tuple((int(i) for i in ws._version.split('.')[:4]))
        while len(version) < 4:
            version += (0,)
        return version

    def _run_simple_model(self, n):
        tmpdir = mkdtemp()
        try:
            from gams import GamsWorkspace, DebugLevel
            ws = GamsWorkspace(debug=DebugLevel.Off, working_directory=tmpdir)
            t1 = ws.add_job_from_string(self._simple_model(n))
            t1.run()
            return True
        except:
            return False
        finally:
            shutil.rmtree(tmpdir)

    def solve(self, *args, **kwds):
        """
        Solve a model via the GAMS Python API.

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
            Keep temporary files. Equivalent of DebugLevel.KeepFiles.
            Summary of temp files can be found in _gams_py_gjo0.pf
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
        """
        self.available()
        from gams import GamsWorkspace, DebugLevel
        try:
            from gams import GamsExceptionExecution
        except ImportError:
            from gams.workspace import GamsExceptionExecution
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
        output_file = StringIO()
        if isinstance(model, IBlock):
            smap_id = model.write(filename=output_file, format=ProblemFormat.gams, _called_by_solver=True, **io_options)
            symbolMap = getattr(model, '._symbol_maps')[smap_id]
        else:
            _, smap_id = model.write(filename=output_file, format=ProblemFormat.gams, io_options=io_options)
            symbolMap = model.solutions.symbol_map[smap_id]
        presolve_completion_time = time.time()
        if report_timing:
            print('      %6.2f seconds required for presolve' % (presolve_completion_time - initial_time))
        newdir = True
        if tmpdir is not None and os.path.exists(tmpdir):
            newdir = False
        ws = GamsWorkspace(debug=DebugLevel.KeepFiles if keepfiles else DebugLevel.Off, working_directory=tmpdir)
        t1 = ws.add_job_from_string(output_file.getvalue())
        try:
            with OutputStream(tee=tee, logfile=logfile) as output_stream:
                t1.run(output=output_stream)
        except GamsExceptionExecution as e:
            try:
                if e.rc == 3:
                    check_expr_evaluation(model, symbolMap, 'direct')
            finally:
                if keepfiles:
                    print('\nGAMS WORKING DIRECTORY: %s\n' % ws.working_directory)
                elif tmpdir is not None:
                    t1 = rec = rec_lo = rec_hi = None
                    file_removal_gams_direct(tmpdir, newdir)
                raise
        except:
            if keepfiles:
                print('\nGAMS WORKING DIRECTORY: %s\n' % ws.working_directory)
            elif tmpdir is not None:
                t1 = rec = rec_lo = rec_hi = None
                file_removal_gams_direct(tmpdir, newdir)
            raise
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
        results.problem.name = os.path.join(ws.working_directory, t1.name + '.gms')
        results.problem.lower_bound = t1.out_db['OBJEST'].find_record().value
        results.problem.upper_bound = t1.out_db['OBJEST'].find_record().value
        results.problem.number_of_variables = t1.out_db['NUMVAR'].find_record().value
        results.problem.number_of_constraints = t1.out_db['NUMEQU'].find_record().value
        results.problem.number_of_nonzeros = t1.out_db['NUMNZ'].find_record().value
        results.problem.number_of_binary_variables = None
        results.problem.number_of_integer_variables = t1.out_db['NUMDVAR'].find_record().value
        results.problem.number_of_continuous_variables = t1.out_db['NUMVAR'].find_record().value - t1.out_db['NUMDVAR'].find_record().value
        results.problem.number_of_objectives = 1
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'Only one objective is allowed.'
        obj = obj[0]
        objctvval = t1.out_db['OBJVAL'].find_record().value
        if obj.is_minimizing():
            results.problem.sense = ProblemSense.minimize
            results.problem.upper_bound = objctvval
        else:
            results.problem.sense = ProblemSense.maximize
            results.problem.lower_bound = objctvval
        results.solver.name = 'GAMS ' + str(self.version())
        results.solver.termination_condition = None
        results.solver.message = None
        solvestat = t1.out_db['SOLVESTAT'].find_record().value
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
        results.solver.return_code = 0
        results.solver.user_time = t1.out_db['ETSOLVE'].find_record().value
        results.solver.system_time = None
        results.solver.wallclock_time = None
        results.solver.termination_message = None
        soln = Solution()
        modelstat = t1.out_db['MODELSTAT'].find_record().value
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
            rec = t1.out_db[sym].find_record()
            soln.variable[sym] = {'Value': rec.level}
            if extract_rc and (not math.isnan(rec.marginal)):
                soln.variable[sym]['rc'] = rec.marginal
        if extract_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed() or not (c.has_lb() or c.has_ub()):
                    continue
                sym = symbolMap.getSymbol(c)
                if c.equality:
                    rec = t1.out_db[sym].find_record()
                    if not math.isnan(rec.marginal):
                        soln.constraint[sym] = {'dual': rec.marginal}
                    else:
                        break
                else:
                    marg = 0
                    if c.lower is not None:
                        rec_lo = t1.out_db[sym + '_lo'].find_record()
                        marg -= rec_lo.marginal
                    if c.upper is not None:
                        rec_hi = t1.out_db[sym + '_hi'].find_record()
                        marg += rec_hi.marginal
                    if not math.isnan(marg):
                        soln.constraint[sym] = {'dual': marg}
                    else:
                        break
        results.solution.insert(soln)
        if keepfiles:
            print('\nGAMS WORKING DIRECTORY: %s\n' % ws.working_directory)
        elif tmpdir is not None:
            t1 = rec = rec_lo = rec_hi = None
            file_removal_gams_direct(tmpdir, newdir)
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