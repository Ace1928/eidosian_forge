from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
def _simplify_gpgp(ex):
    components = ex.components
    a = []
    comp_map = []
    for i, comp in enumerate(components):
        comp_map.extend([i] * comp.rank)
    dum = [(i[0], i[1], comp_map[i[0]], comp_map[i[1]]) for i in ex.dum]
    for i in range(len(components)):
        if components[i] != GammaMatrix:
            continue
        for dx in dum:
            if dx[2] == i:
                p_pos1 = dx[3]
            elif dx[3] == i:
                p_pos1 = dx[2]
            else:
                continue
            comp1 = components[p_pos1]
            if comp1.comm == 0 and comp1.rank == 1:
                a.append((i, p_pos1))
    if not a:
        return ex
    elim = set()
    tv = []
    hit = True
    coeff = S.One
    ta = None
    while hit:
        hit = False
        for i, ai in enumerate(a[:-1]):
            if ai[0] in elim:
                continue
            if ai[0] != a[i + 1][0] - 1:
                continue
            if components[ai[1]] != components[a[i + 1][1]]:
                continue
            elim.add(ai[0])
            elim.add(ai[1])
            elim.add(a[i + 1][0])
            elim.add(a[i + 1][1])
            if not ta:
                ta = ex.split()
                mu = TensorIndex('mu', LorentzIndex)
            hit = True
            if i == 0:
                coeff = ex.coeff
            tx = components[ai[1]](mu) * components[ai[1]](-mu)
            if len(a) == 2:
                tx *= 4
            tv.append(tx)
            break
    if tv:
        a = [x for j, x in enumerate(ta) if j not in elim]
        a.extend(tv)
        t = tensor_mul(*a) * coeff
        return t
    else:
        return ex