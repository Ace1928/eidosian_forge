import logging
from pyomo.common.deprecation import deprecated
from pyomo.core.base import (
from pyomo.duality.collect import collect_linear_terms
@TransformationFactory.register('duality.linear_dual', doc='[DEPRECATED] Dualize a linear model')
@deprecated('Use of the pyomo.duality package is deprecated. There are known bugs in pyomo.duality, and we do not recommend the use of this code. Development of dualization capabilities has been shifted to the Pyomo Adversarial Optimization (PAO) library. Please contact William Hart for further details (wehart@sandia.gov).', version='5.6.2')
class LinearDual_PyomoTransformation(Transformation):

    def __init__(self):
        super(LinearDual_PyomoTransformation, self).__init__()

    def _create_using(self, instance, **kwds):
        options = kwds.pop('options', {})
        bname = options.get('block', None)
        block = None
        if block is None:
            block = instance
        else:
            for name, data in instance.component_map(Block, active=True).items():
                if name == bname:
                    block = instance
        if block is None:
            raise RuntimeError('Missing block: ' + bname)
        instance_ = self._dualize(block)
        return instance_

    def _dualize(self, block, unfixed=[]):
        """
        Generate the dual of a block
        """
        A, b_coef, c_rhs, c_sense, d_sense, vnames, cnames, v_domain = collect_linear_terms(block, unfixed)
        if isinstance(block, Model):
            dual = ConcreteModel()
        else:
            dual = Block()
        dual.construct()
        _vars = {}

        def getvar(name, ndx=None):
            v = _vars.get((name, ndx), None)
            if v is None:
                v = Var()
                if ndx is None:
                    v_name = name
                elif type(ndx) is tuple:
                    v_name = '%s[%s]' % (name, ','.join(map(str, ndx)))
                else:
                    v_name = '%s[%s]' % (name, str(ndx))
                setattr(dual, v_name, v)
                _vars[name, ndx] = v
            return v
        if d_sense == minimize:
            dual.o = Objective(expr=sum((-b_coef[name, ndx] * getvar(name, ndx) for name, ndx in b_coef)), sense=d_sense)
        else:
            dual.o = Objective(expr=sum((b_coef[name, ndx] * getvar(name, ndx) for name, ndx in b_coef)), sense=d_sense)
        for cname in A:
            for ndx, terms in A[cname].items():
                expr = 0
                for term in terms:
                    expr += term.coef * getvar(term.var, term.ndx)
                if not (cname, ndx) in c_rhs:
                    c_rhs[cname, ndx] = 0.0
                if c_sense[cname, ndx] == 'e':
                    e = expr - c_rhs[cname, ndx] == 0
                elif c_sense[cname, ndx] == 'l':
                    e = expr - c_rhs[cname, ndx] <= 0
                else:
                    e = expr - c_rhs[cname, ndx] >= 0
                c = Constraint(expr=e)
                if ndx is None:
                    c_name = cname
                elif type(ndx) is tuple:
                    c_name = '%s[%s]' % (cname, ','.join(map(str, ndx)))
                else:
                    c_name = '%s[%s]' % (cname, str(ndx))
                setattr(dual, c_name, c)
            for (name, ndx), domain in v_domain.items():
                v = getvar(name, ndx)
                flag = type(ndx) is tuple and (ndx[-1] == 'lb' or ndx[-1] == 'ub')
                if domain == 1:
                    v.domain = NonNegativeReals
                elif domain == -1:
                    v.domain = NonPositiveReals
                else:
                    v.domain = Reals
        return dual