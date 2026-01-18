import time
import logging
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
Solve the problem