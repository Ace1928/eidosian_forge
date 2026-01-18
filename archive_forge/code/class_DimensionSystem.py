from __future__ import annotations
import collections
from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow
class DimensionSystem(Basic, _QuantityMapper):
    """
    DimensionSystem represents a coherent set of dimensions.

    The constructor takes three parameters:

    - base dimensions;
    - derived dimensions: these are defined in terms of the base dimensions
      (for example velocity is defined from the division of length by time);
    - dependency of dimensions: how the derived dimensions depend
      on the base dimensions.

    Optionally either the ``derived_dims`` or the ``dimensional_dependencies``
    may be omitted.
    """

    def __new__(cls, base_dims, derived_dims=(), dimensional_dependencies={}):
        dimensional_dependencies = dict(dimensional_dependencies)

        def parse_dim(dim):
            if isinstance(dim, str):
                dim = Dimension(Symbol(dim))
            elif isinstance(dim, Dimension):
                pass
            elif isinstance(dim, Symbol):
                dim = Dimension(dim)
            else:
                raise TypeError('%s wrong type' % dim)
            return dim
        base_dims = [parse_dim(i) for i in base_dims]
        derived_dims = [parse_dim(i) for i in derived_dims]
        for dim in base_dims:
            if dim in dimensional_dependencies and (len(dimensional_dependencies[dim]) != 1 or dimensional_dependencies[dim].get(dim, None) != 1):
                raise IndexError('Repeated value in base dimensions')
            dimensional_dependencies[dim] = Dict({dim: 1})

        def parse_dim_name(dim):
            if isinstance(dim, Dimension):
                return dim
            elif isinstance(dim, str):
                return Dimension(Symbol(dim))
            elif isinstance(dim, Symbol):
                return Dimension(dim)
            else:
                raise TypeError('unrecognized type %s for %s' % (type(dim), dim))
        for dim in dimensional_dependencies.keys():
            dim = parse_dim(dim)
            if dim not in derived_dims and dim not in base_dims:
                derived_dims.append(dim)

        def parse_dict(d):
            return Dict({parse_dim_name(i): j for i, j in d.items()})
        dimensional_dependencies = {parse_dim_name(i): parse_dict(j) for i, j in dimensional_dependencies.items()}
        for dim in derived_dims:
            if dim in base_dims:
                raise ValueError('Dimension %s both in base and derived' % dim)
            if dim not in dimensional_dependencies:
                dimensional_dependencies[dim] = Dict({dim: 1})
        base_dims.sort(key=default_sort_key)
        derived_dims.sort(key=default_sort_key)
        base_dims = Tuple(*base_dims)
        derived_dims = Tuple(*derived_dims)
        dimensional_dependencies = Dict({i: Dict(j) for i, j in dimensional_dependencies.items()})
        obj = Basic.__new__(cls, base_dims, derived_dims, dimensional_dependencies)
        return obj

    @property
    def base_dims(self):
        return self.args[0]

    @property
    def derived_dims(self):
        return self.args[1]

    @property
    def dimensional_dependencies(self):
        return self.args[2]

    def _get_dimensional_dependencies_for_name(self, dimension):
        if isinstance(dimension, str):
            dimension = Dimension(Symbol(dimension))
        elif not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)
        if dimension.name.is_Symbol:
            return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))
        if dimension.name.is_number or dimension.name.is_NumberSymbol:
            return {}
        get_for_name = self._get_dimensional_dependencies_for_name
        if dimension.name.is_Mul:
            ret = collections.defaultdict(int)
            dicts = [get_for_name(i) for i in dimension.name.args]
            for d in dicts:
                for k, v in d.items():
                    ret[k] += v
            return {k: v for k, v in ret.items() if v != 0}
        if dimension.name.is_Add:
            dicts = [get_for_name(i) for i in dimension.name.args]
            if all((d == dicts[0] for d in dicts[1:])):
                return dicts[0]
            raise TypeError('Only equivalent dimensions can be added or subtracted.')
        if dimension.name.is_Pow:
            dim_base = get_for_name(dimension.name.base)
            dim_exp = get_for_name(dimension.name.exp)
            if dim_exp == {} or dimension.name.exp.is_Symbol:
                return {k: v * dimension.name.exp for k, v in dim_base.items()}
            else:
                raise TypeError('The exponent for the power operator must be a Symbol or dimensionless.')
        if dimension.name.is_Function:
            args = (Dimension._from_dimensional_dependencies(get_for_name(arg)) for arg in dimension.name.args)
            result = dimension.name.func(*args)
            dicts = [get_for_name(i) for i in dimension.name.args]
            if isinstance(result, Dimension):
                return self.get_dimensional_dependencies(result)
            elif result.func == dimension.name.func:
                if isinstance(dimension.name, TrigonometricFunction):
                    if dicts[0] in ({}, {Dimension('angle'): 1}):
                        return {}
                    else:
                        raise TypeError('The input argument for the function {} must be dimensionless or have dimensions of angle.'.format(dimension.func))
                elif all((item == {} for item in dicts)):
                    return {}
                else:
                    raise TypeError('The input arguments for the function {} must be dimensionless.'.format(dimension.func))
            else:
                return get_for_name(result)
        raise TypeError('Type {} not implemented for get_dimensional_dependencies'.format(type(dimension.name)))

    def get_dimensional_dependencies(self, name, mark_dimensionless=False):
        dimdep = self._get_dimensional_dependencies_for_name(name)
        if mark_dimensionless and dimdep == {}:
            return {Dimension(1): 1}
        return {k: v for k, v in dimdep.items()}

    def equivalent_dims(self, dim1, dim2):
        deps1 = self.get_dimensional_dependencies(dim1)
        deps2 = self.get_dimensional_dependencies(dim2)
        return deps1 == deps2

    def extend(self, new_base_dims, new_derived_dims=(), new_dim_deps=None):
        deps = dict(self.dimensional_dependencies)
        if new_dim_deps:
            deps.update(new_dim_deps)
        new_dim_sys = DimensionSystem(tuple(self.base_dims) + tuple(new_base_dims), tuple(self.derived_dims) + tuple(new_derived_dims), deps)
        new_dim_sys._quantity_dimension_map.update(self._quantity_dimension_map)
        new_dim_sys._quantity_scale_factors.update(self._quantity_scale_factors)
        return new_dim_sys

    def is_dimensionless(self, dimension):
        """
        Check if the dimension object really has a dimension.

        A dimension should have at least one component with non-zero power.
        """
        if dimension.name == 1:
            return True
        return self.get_dimensional_dependencies(dimension) == {}

    @property
    def list_can_dims(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        List all canonical dimension names.
        """
        dimset = set()
        for i in self.base_dims:
            dimset.update(set(self.get_dimensional_dependencies(i).keys()))
        return tuple(sorted(dimset, key=str))

    @property
    def inv_can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Compute the inverse transformation matrix from the base to the
        canonical dimension basis.

        It corresponds to the matrix where columns are the vector of base
        dimensions in canonical basis.

        This matrix will almost never be used because dimensions are always
        defined with respect to the canonical basis, so no work has to be done
        to get them in this basis. Nonetheless if this matrix is not square
        (or not invertible) it means that we have chosen a bad basis.
        """
        matrix = reduce(lambda x, y: x.row_join(y), [self.dim_can_vector(d) for d in self.base_dims])
        return matrix

    @property
    def can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Return the canonical transformation matrix from the canonical to the
        base dimension basis.

        It is the inverse of the matrix computed with inv_can_transf_matrix().
        """
        return reduce(lambda x, y: x.row_join(y), [self.dim_can_vector(d) for d in sorted(self.base_dims, key=str)]).inv()

    def dim_can_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """
        vec = []
        for d in self.list_can_dims:
            vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
        return Matrix(vec)

    def dim_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.


        Vector representation in terms of the base dimensions.
        """
        return self.can_transf_matrix * Matrix(self.dim_can_vector(dim))

    def print_dim_base(self, dim):
        """
        Give the string expression of a dimension in term of the basis symbols.
        """
        dims = self.dim_vector(dim)
        symbols = [i.symbol if i.symbol is not None else i.name for i in self.base_dims]
        res = S.One
        for s, p in zip(symbols, dims):
            res *= s ** p
        return res

    @property
    def dim(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Give the dimension of the system.

        That is return the number of dimensions forming the basis.
        """
        return len(self.base_dims)

    @property
    def is_consistent(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Check if the system is well defined.
        """
        return self.inv_can_transf_matrix.is_square