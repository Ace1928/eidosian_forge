from collections import OrderedDict, defaultdict
from itertools import chain
from chempy.kinetics.ode import get_odesys
from chempy.units import to_unitless, linspace, logspace_from_lin
def _C(k):
    return to_unitless(c0[k], output_conc_unit)