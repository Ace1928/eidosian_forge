from functools import reduce
from collections.abc import Iterable
from typing import Optional
from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift
def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
    from sympy.matrices.dense import Matrix
    dimension_system = unit_system.get_dimension_system()
    expr_dim = Dimension(unit_system.get_dimensional_expr(expr))
    dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
    target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]
    canon_dim_units = [i for x in target_dims for i in dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]
    canon_expr_units = set(dim_dependencies)
    if not canon_expr_units.issubset(set(canon_dim_units)):
        return None
    seen = set()
    canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]
    camat = Matrix([[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims] for j in canon_dim_units])
    exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])
    try:
        res_exponents = camat.solve(exprmat)
    except NonInvertibleMatrixError:
        return None
    return res_exponents