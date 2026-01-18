import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def _make_model_with_external_functions(self):
    m = pyo.ConcreteModel()
    gsl = find_GSL()
    m.bessel = pyo.ExternalFunction(library=gsl, function='gsl_sf_bessel_J0')
    m.fermi = pyo.ExternalFunction(library=gsl, function='gsl_sf_fermi_dirac_m1')
    m.v1 = pyo.Var(initialize=1.0)
    m.v2 = pyo.Var(initialize=2.0)
    m.v3 = pyo.Var(initialize=3.0)
    m.con1 = pyo.Constraint(expr=m.v1 == 0.5)
    m.con2 = pyo.Constraint(expr=2 * m.fermi(m.v1) + m.v2 ** 2 - m.v3 == 1.0)
    m.con3 = pyo.Constraint(expr=m.bessel(m.v1) - m.bessel(m.v2) + m.v3 ** 2 == 2.0)
    return m