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
@SolverFactory.register('gams', doc='The GAMS modeling language')
class GAMSSolver(_GAMSSolver):
    """
    A generic interface to GAMS solvers.

    Pass solver_io keyword arg to SolverFactory to choose solver mode:
        solver_io='direct' or 'python' to use GAMS Python API
            Requires installation, visit Python API page on gams.com for help.
        solver_io='shell' or 'gms' to use command line to call gams
            Requires the gams executable be on your system PATH.
    """

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'shell')
        if mode is None:
            mode = 'shell'
        if mode == 'direct' or mode == 'python':
            return SolverFactory('_gams_direct', **kwds)
        if mode == 'shell' or mode == 'gms':
            return SolverFactory('_gams_shell', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return