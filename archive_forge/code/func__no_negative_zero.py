import logging
from io import StringIO
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def _no_negative_zero(val):
    """Make sure -0 is never output. Makes diff tests easier."""
    if val == 0:
        return 0
    return val