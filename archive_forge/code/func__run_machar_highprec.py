from numpy.core._machar import MachAr
import numpy.core.numerictypes as ntypes
from numpy import errstate, array
def _run_machar_highprec(self):
    try:
        hiprec = ntypes.float96
        MachAr(lambda v: array(v, hiprec))
    except AttributeError:
        'Skipping test: no ntypes.float96 available on this platform.'