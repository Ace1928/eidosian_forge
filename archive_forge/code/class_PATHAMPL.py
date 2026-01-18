import logging
from pyomo.opt.base.solvers import SolverFactory
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.solvers.plugins.solvers.ASL import ASL
@SolverFactory.register('path', doc='Nonlinear MCP solver')
class PATHAMPL(ASL):
    """An interface to the PATH MCP solver."""

    def __init__(self, **kwds):
        kwds['type'] = 'path'
        ASL.__init__(self, **kwds)
        self._metasolver = False
        self._capabilities = Bunch()
        self._capabilities.linear = True

    def _default_executable(self):
        executable = Executable('pathampl')
        if not executable:
            logger.warning("Could not locate the 'pathampl' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def create_command_line(self, executable, problem_files):
        self.options.solver = 'pathampl'
        return ASL.create_command_line(self, executable, problem_files)