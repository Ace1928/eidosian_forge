from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
from .. import constants
import warnings
import sys
import re
def _ismip(lp):
    """Check whether lp is a MIP.

    From an XPRESS point of view, a problem is also a MIP if it contains
    SOS constraints."""
    return lp.isMIP() or len(lp.sos1) or len(lp.sos2)