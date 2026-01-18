from pyomo.core.expr import ProductExpression, PowExpression
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Binary, value
from pyomo.core.base import (
from pyomo.core.base.var import _VarData
import logging
def _create_using(self, model, **kwds):
    precision = kwds.pop('precision', 8)
    user_discretize = kwds.pop('discretize', set())
    verbose = kwds.pop('verbose', False)
    M = model.clone()
    _discretize = {}
    if user_discretize:
        for _var in user_discretize:
            _v = M.find_component(_var.name)
            if _v.component() is _v:
                for _vv in _v.itervalues():
                    _discretize.setdefault(id(_vv), len(_discretize))
            else:
                _discretize.setdefault(id(_v), len(_discretize))
    bilinear_terms = []
    quadratic_terms = []
    for constraint in M.component_map(Constraint, active=True).itervalues():
        for cname, c in constraint._data.iteritems():
            if c.body.polynomial_degree() != 2:
                continue
            self._collect_bilinear(c.body, bilinear_terms, quadratic_terms)
    _counts = {}
    for q in quadratic_terms:
        if not q[1].is_continuous():
            continue
        _id = id(q[1])
        if _id not in _counts:
            _counts[_id] = (q[1], set())
        _counts[_id][1].add(_id)
    for bi in bilinear_terms:
        for i in (0, 1):
            if not bi[i + 1].is_continuous():
                continue
            _id = id(bi[i + 1])
            if _id not in _counts:
                _counts[_id] = (bi[i + 1], set())
            _counts[_id][1].add(id(bi[2 - i]))
    _tmp_counts = dict(_counts)
    for _id in _discretize:
        for _i in _tmp_counts[_id][1]:
            if _i == _id:
                continue
            _tmp_counts[_i][1].remove(_id)
        del _tmp_counts[_id]
    while _tmp_counts:
        _ct, _id = max(((len(_tmp_counts[i][1]), i) for i in _tmp_counts))
        if not _ct:
            break
        _discretize.setdefault(_id, len(_discretize))
        for _i in list(_tmp_counts[_id][1]):
            if _i == _id:
                continue
            _tmp_counts[_i][1].remove(_id)
        del _tmp_counts[_id]
    if False:
        M._radix_linearization = Block()
        _block = M._radix_linearization
    else:
        _block = M
    _block.DISCRETIZATION = RangeSet(precision)
    _block.DISCRETIZED_VARIABLES = RangeSet(0, len(_discretize) - 1)
    _block.z = Var(_block.DISCRETIZED_VARIABLES, _block.DISCRETIZATION, within=Binary)
    _block.dv = Var(_block.DISCRETIZED_VARIABLES, bounds=(0, 2 ** (-precision)))
    for _id, _idx in _discretize.items():
        if verbose:
            logger.info('Discretizing variable %s as %s' % (_counts[_id][0].name, _idx))
        self._discretize_variable(_block, _counts[_id][0], _idx)
    _known_bilinear = {}
    for _expr, _x1, _x2 in bilinear_terms:
        self._discretize_term(_expr, _x1, _x2, _block, _discretize, _known_bilinear)
    return M