from pyomo.core.expr import ProductExpression, PowExpression
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Binary, value
from pyomo.core.base import (
from pyomo.core.base.var import _VarData
import logging
def _discretize_bilinear(self, b, v, v_idx, u, u_idx):
    _z = b.z
    _dv = b.dv[v_idx]
    _u = Var(b.DISCRETIZATION, within=u.domain, bounds=u.bounds)
    logger.info('Discretizing (v=%s)*(u=%s) as u%s_v%s' % (v.name, u.name, u_idx, v_idx))
    b.add_component('u%s_v%s' % (u_idx, v_idx), _u)
    _lb, _ub = u.bounds
    if _lb is None or _ub is None:
        raise RuntimeError("Couldn't relax variable %s: missing finite lower/upper bounds." % u.name)
    _c = ConstraintList()
    b.add_component('c_disaggregate_u%s_v%s' % (u_idx, v_idx), _c)
    for k in b.DISCRETIZATION:
        _c.add(expr=_lb * _z[v_idx, k] <= _u[k])
        _c.add(expr=_u[k] <= _ub * _z[v_idx, k])
        _c.add(expr=_lb * (1 - _z[v_idx, k]) <= u - _u[k])
        _c.add(expr=u - _u[k] <= _ub * (1 - _z[v_idx, k]))
    _v_lb, _v_ub = v.bounds
    _bnd_rng = (_v_lb * _lb, _v_lb * _ub, _v_ub * _lb, _v_ub * _ub)
    _w = Var(bounds=(min(_bnd_rng), max(_bnd_rng)))
    b.add_component('w%s_v%s' % (u_idx, v_idx), _w)
    K = max(b.DISCRETIZATION)
    _dw = Var(bounds=(min(0, _lb * 2 ** (-K), _ub * 2 ** (-K)), max(0, _lb * 2 ** (-K), _ub * 2 ** (-K))))
    b.add_component('dw%s_v%s' % (u_idx, v_idx), _dw)
    _c = Constraint(expr=_w == _v_lb * u + (_v_ub - _v_lb) * (sum((2 ** (-k) * _u[k] for k in b.DISCRETIZATION)) + _dw))
    b.add_component('c_bilinear_u%s_v%s' % (u_idx, v_idx), _c)
    _c = ConstraintList()
    b.add_component('c_mccormick_u%s_v%s' % (u_idx, v_idx), _c)
    _c.add(expr=_lb * _dv <= _dw)
    _c.add(expr=_dw <= _ub * _dv)
    _c.add(expr=(u - _ub) * 2 ** (-K) + _ub * _dv <= _dw)
    _c.add(expr=_dw <= (u - _lb) * 2 ** (-K) + _lb * _dv)
    return _w