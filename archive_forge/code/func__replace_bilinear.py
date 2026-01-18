import logging
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
def _replace_bilinear(self, expr, instance):
    idMap = {}
    terms = generate_standard_repn(expr, idMap=idMap)
    e = terms.constant
    for var, coef in zip(terms.linear_vars, terms.linear_coefs):
        e += coef * var
    if len(terms.quadratic_coefs) > 0:
        for vars_, coef_ in zip(terms.quadratic_vars, terms.quadratic_coefs):
            if vars_[0].is_binary():
                v = instance.bilinear_data_.cache.get((id(vars_[0]), id(vars_[1])), None)
                if v is None:
                    instance.bilinear_data_.vlist_boolean.append(vars_[0])
                    v = instance.bilinear_data_.vlist.add()
                    instance.bilinear_data_.cache[id(vars_[0]), id(vars_[1])] = v
                    bounds = vars_[1].bounds
                    v.setlb(bounds[0])
                    v.setub(bounds[1])
                    id_ = len(instance.bilinear_data_.vlist)
                    instance.bilinear_data_.IDX.add(id_)
                    d0 = instance.bilinear_data_.disjuncts_[id_, 0]
                    d0.c1 = Constraint(expr=vars_[0] == 1)
                    d0.c2 = Constraint(expr=v == vars_[1])
                    d1 = instance.bilinear_data_.disjuncts_[id_, 1]
                    d1.c1 = Constraint(expr=vars_[0] == 0)
                    d1.c2 = Constraint(expr=v == 0)
                    instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_, 0], instance.bilinear_data_.disjuncts_[id_, 1]]
                    instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_, 0], instance.bilinear_data_.disjuncts_[id_, 1]]
                e += coef_ * v
            elif vars_[1].is_binary():
                v = instance.bilinear_data_.cache.get((id(vars_[1]), id(vars_[0])), None)
                if v is None:
                    instance.bilinear_data_.vlist_boolean.append(vars_[1])
                    v = instance.bilinear_data_.vlist.add()
                    instance.bilinear_data_.cache[id(vars_[1]), id(vars_[0])] = v
                    bounds = vars_[0].bounds
                    v.setlb(bounds[0])
                    v.setub(bounds[1])
                    id_ = len(instance.bilinear_data_.vlist)
                    instance.bilinear_data_.IDX.add(id_)
                    d0 = instance.bilinear_data_.disjuncts_[id_, 0]
                    d0.c1 = Constraint(expr=vars_[1] == 1)
                    d0.c2 = Constraint(expr=v == vars_[0])
                    d1 = instance.bilinear_data_.disjuncts_[id_, 1]
                    d1.c1 = Constraint(expr=vars_[1] == 0)
                    d1.c2 = Constraint(expr=v == 0)
                    instance.bilinear_data_.disjunction_data[id_] = [instance.bilinear_data_.disjuncts_[id_, 0], instance.bilinear_data_.disjuncts_[id_, 1]]
                e += coef_ * v
            else:
                e += coef_ * vars_[0] * vars_[1]
    return e