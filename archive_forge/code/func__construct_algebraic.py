from math import prod
from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public
def _construct_algebraic(coeffs, opt):
    """We know that coefficients are algebraic so construct the extension. """
    from sympy.polys.numberfields import primitive_element
    exts = set()

    def build_trees(args):
        trees = []
        for a in args:
            if a.is_Rational:
                tree = ('Q', QQ.from_sympy(a))
            elif a.is_Add:
                tree = ('+', build_trees(a.args))
            elif a.is_Mul:
                tree = ('*', build_trees(a.args))
            else:
                tree = ('e', a)
                exts.add(a)
            trees.append(tree)
        return trees
    trees = build_trees(coeffs)
    exts = list(ordered(exts))
    g, span, H = primitive_element(exts, ex=True, polys=True)
    root = sum([s * ext for s, ext in zip(span, exts)])
    domain, g = (QQ.algebraic_field((g, root)), g.rep.rep)
    exts_dom = [domain.dtype.from_list(h, g, QQ) for h in H]
    exts_map = dict(zip(exts, exts_dom))

    def convert_tree(tree):
        op, args = tree
        if op == 'Q':
            return domain.dtype.from_list([args], g, QQ)
        elif op == '+':
            return sum((convert_tree(a) for a in args), domain.zero)
        elif op == '*':
            return prod((convert_tree(a) for a in args))
        elif op == 'e':
            return exts_map[args]
        else:
            raise RuntimeError
    result = [convert_tree(tree) for tree in trees]
    return (domain, result)