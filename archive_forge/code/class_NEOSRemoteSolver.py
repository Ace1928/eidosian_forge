import logging
from pyomo.opt.base import SolverFactory, ProblemFormat, ResultsFormat
from pyomo.opt.solver import SystemCallSolver
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
@SolverFactory.register('_neos', 'Interface for solvers hosted on NEOS')
class NEOSRemoteSolver(SystemCallSolver):
    """A wrapper class for NEOS Remote Solvers"""

    def __init__(self, **kwds):
        kwds['type'] = 'neos'
        SystemCallSolver.__init__(self, **kwds)
        self._valid_problem_formats = [ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self._problem_format = ProblemFormat.nl
        self._results_format = ResultsFormat.sol

    def create_command_line(self, executable, problem_files):
        """
        Create the local *.sol and *.log files, which will be
        populated by NEOS.
        """
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.neos.log')
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix='.neos.sol')
            self._results_file = self._soln_file
        if self._keepfiles:
            if self._log_file is not None:
                logger.info("Solver log file: '%s'" % (self._log_file,))
            if self._soln_file is not None:
                logger.info("Solver solution file: '%s'" % (self._soln_file,))
            if self._problem_files != []:
                logger.info('Solver problem files: %s' % (self._problem_files,))
        return Bunch(cmd='', log_file=self._log_file, env='')

    def _default_executable(self):
        return True