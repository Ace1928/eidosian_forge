import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _initcon(m):
    yield (m.Ca[m.time.first()] == m.Ca0)
    yield (m.Cb[m.time.first()] == m.Cb0)
    yield (m.Cc[m.time.first()] == m.Cc0)
    yield (m.Vr[m.time.first()] == m.Vr0)
    yield (m.Tr[m.time.first()] == m.Tr0)