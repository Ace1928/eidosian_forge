from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import Executable
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.writers import NLWriter
from pyomo.common.log import LogStream
import logging
import subprocess
from pyomo.core.kernel.objective import minimize
import math
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.visitor import replace_expressions
from typing import Optional, Sequence, NoReturn, List, Mapping
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import TeeStream
import sys
from typing import Dict
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
import os
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager
class IpoptConfig(SolverConfig):

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super(IpoptConfig, self).__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.declare('executable', ConfigValue())
        self.declare('filename', ConfigValue(domain=str))
        self.declare('keepfiles', ConfigValue(domain=bool))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))
        self.executable = Executable('ipopt')
        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO