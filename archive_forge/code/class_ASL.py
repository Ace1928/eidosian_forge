import os
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.solver import SystemCallSolver
from pyomo.core.kernel.block import IBlock
from pyomo.solvers.mockmip import MockMIP
from pyomo.core import TransformationFactory
import logging
@SolverFactory.register('asl', doc='Interface for solvers using the AMPL Solver Library')
class ASL(SystemCallSolver):
    """A generic optimizer that uses the AMPL Solver Library to interface with applications."""

    def __init__(self, **kwds):
        if not 'type' in kwds:
            kwds['type'] = 'asl'
        SystemCallSolver.__init__(self, **kwds)
        self._metasolver = True
        self._valid_problem_formats = [ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self.set_problem_format(ProblemFormat.nl)
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.sol

    def _default_executable(self):
        if self.options.solver is None:
            logger.warning('No solver option specified for ASL solver interface')
            return None
        if not self.options.solver:
            logger.warning('No solver option specified for ASL solver interface')
            return None
        executable = Executable(self.options.solver)
        if not executable:
            logger.warning("Could not locate the '%s' executable, which is required for solver %s" % (self.options.solver, self.name))
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        try:
            results = subprocess.run([solver_exec, '-v'], timeout=5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            ver = _extract_version(results.stdout)
            if ver is None:
                if results.stdout.strip().split()[-1].startswith('ASL('):
                    return '0.0.0'
            return ver
        except OSError:
            pass
        except subprocess.TimeoutExpired:
            pass

    def available(self, exception_flag=True):
        if not super().available(exception_flag):
            return False
        return self.version() is not None

    def create_command_line(self, executable, problem_files):
        assert self._problem_format == ProblemFormat.nl
        assert self._results_format == ResultsFormat.sol
        solver_name = os.path.basename(self.options.solver)
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='_%s.log' % solver_name)
        if self._soln_file is not None:
            logger.warning("The 'soln_file' keyword will be ignored for solver=" + self.type)
        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            fname = '.'.join(tmp[:-1])
        self._soln_file = fname + '.sol'
        self._results_file = self._soln_file
        env = os.environ.copy()
        if 'PYOMO_AMPLFUNC' in env:
            if 'AMPLFUNC' in env:
                env['AMPLFUNC'] += '\n' + env['PYOMO_AMPLFUNC']
            else:
                env['AMPLFUNC'] = env['PYOMO_AMPLFUNC']
        cmd = [executable, problem_files[0], '-AMPL']
        if self._timer:
            cmd.insert(0, self._timer)
        opt = []
        for key in self.options:
            if key == 'solver':
                continue
            if isinstance(self.options[key], str) and ' ' in self.options[key]:
                opt.append(key + '="' + str(self.options[key]) + '"')
                cmd.append(str(key) + '=' + str(self.options[key]))
            elif key == 'subsolver':
                opt.append('solver=' + str(self.options[key]))
                cmd.append(str(key) + '=' + str(self.options[key]))
            else:
                opt.append(key + '=' + str(self.options[key]))
                cmd.append(str(key) + '=' + str(self.options[key]))
        envstr = '%s_options' % self.options.solver
        env[envstr] = ' '.join(opt)
        return Bunch(cmd=cmd, log_file=self._log_file, env=env)

    def _presolve(self, *args, **kwds):
        if not isinstance(args[0], str) and (not isinstance(args[0], IBlock)):
            self._instance = args[0]
            xfrm = TransformationFactory('mpec.nl')
            xfrm.apply_to(self._instance)
            if len(self._instance._transformation_data['mpec.nl'].compl_cuids) == 0:
                self._instance = None
            else:
                args = (self._instance,)
        else:
            self._instance = None
        SystemCallSolver._presolve(self, *args, **kwds)

    def _postsolve(self):
        mpec = False
        if not self._instance is None:
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.nl'].compl_cuids:
                mpec = True
                cobj = cuid.find_component_on(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
        self._instance = None
        return SystemCallSolver._postsolve(self)