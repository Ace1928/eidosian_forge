import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def _IN(m, i):
    return {'pressure': pipe.pIn, 'flow': pipe.composition[i] * pipe.flow}