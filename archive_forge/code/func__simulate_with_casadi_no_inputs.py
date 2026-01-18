import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _simulate_with_casadi_no_inputs(self, initcon, tsim, integrator, integrator_options):
    xalltemp = [self._templatemap[i] for i in self._diffvars]
    xall = casadi.vertcat(*xalltemp)
    odealltemp = [convert_pyomo2casadi(self._rhsdict[i]) for i in self._derivlist]
    odeall = casadi.vertcat(*odealltemp)
    dae = {'x': xall, 'ode': odeall}
    if len(self._algvars) != 0:
        zalltemp = [self._templatemap[i] for i in self._simalgvars]
        zall = casadi.vertcat(*zalltemp)
        algalltemp = [convert_pyomo2casadi(i) for i in self._alglist]
        algall = casadi.vertcat(*algalltemp)
        dae['z'] = zall
        dae['alg'] = algall
    integrator_options['grid'] = tsim
    integrator_options['output_t0'] = True
    F = casadi.integrator('F', integrator, dae, integrator_options)
    sol = F(x0=initcon)
    profile = sol['xf'].full().T
    if len(self._algvars) != 0:
        algprofile = sol['zf'].full().T
        profile = np.concatenate((profile, algprofile), axis=1)
    return [tsim, profile]