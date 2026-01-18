from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
class TensorHead(Basic):
    """
    Tensor head of the tensor.

    Parameters
    ==========

    name : name of the tensor
    index_types : list of TensorIndexType
    symmetry : TensorSymmetry of the tensor
    comm : commutation group number

    Attributes
    ==========

    ``name``
    ``index_types``
    ``rank`` : total number of indices
    ``symmetry``
    ``comm`` : commutation group

    Notes
    =====

    Similar to ``symbols`` multiple TensorHeads can be created using
    ``tensorhead(s, typ, sym=None, comm=0)`` function, where ``s``
    is the string of names and ``sym`` is the monoterm tensor symmetry
    (see ``tensorsymmetry``).

    A ``TensorHead`` belongs to a commutation group, defined by a
    symbol on number ``comm`` (see ``_TensorManager.set_comm``);
    tensors in a commutation group have the same commutation properties;
    by default ``comm`` is ``0``, the group of the commuting tensors.

    Examples
    ========

    Define a fully antisymmetric tensor of rank 2:

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> asym2 = TensorSymmetry.fully_symmetric(-2)
    >>> A = TensorHead('A', [Lorentz, Lorentz], asym2)

    Examples with ndarray values, the components data assigned to the
    ``TensorHead`` object are assumed to be in a fully-contravariant
    representation. In case it is necessary to assign components data which
    represents the values of a non-fully covariant tensor, see the other
    examples.

    >>> from sympy.tensor.tensor import tensor_indices
    >>> from sympy import diag
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i0, i1 = tensor_indices('i0:2', Lorentz)

    Specify a replacement dictionary to keep track of the arrays to use for
    replacements in the tensorial expression. The ``TensorIndexType`` is
    associated to the metric used for contractions (in fully covariant form):

    >>> repl = {Lorentz: diag(1, -1, -1, -1)}

    Let's see some examples of working with components with the electromagnetic
    tensor:

    >>> from sympy import symbols
    >>> Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
    >>> c = symbols('c', positive=True)

    Let's define `F`, an antisymmetric tensor:

    >>> F = TensorHead('F', [Lorentz, Lorentz], asym2)

    Let's update the dictionary to contain the matrix to use in the
    replacements:

    >>> repl.update({F(-i0, -i1): [
    ... [0, Ex/c, Ey/c, Ez/c],
    ... [-Ex/c, 0, -Bz, By],
    ... [-Ey/c, Bz, 0, -Bx],
    ... [-Ez/c, -By, Bx, 0]]})

    Now it is possible to retrieve the contravariant form of the Electromagnetic
    tensor:

    >>> F(i0, i1).replace_with_arrays(repl, [i0, i1])
    [[0, -E_x/c, -E_y/c, -E_z/c], [E_x/c, 0, -B_z, B_y], [E_y/c, B_z, 0, -B_x], [E_z/c, -B_y, B_x, 0]]

    and the mixed contravariant-covariant form:

    >>> F(i0, -i1).replace_with_arrays(repl, [i0, -i1])
    [[0, E_x/c, E_y/c, E_z/c], [E_x/c, 0, B_z, -B_y], [E_y/c, -B_z, 0, B_x], [E_z/c, B_y, -B_x, 0]]

    Energy-momentum of a particle may be represented as:

    >>> from sympy import symbols
    >>> P = TensorHead('P', [Lorentz], TensorSymmetry.no_symmetry(1))
    >>> E, px, py, pz = symbols('E p_x p_y p_z', positive=True)
    >>> repl.update({P(i0): [E, px, py, pz]})

    The contravariant and covariant components are, respectively:

    >>> P(i0).replace_with_arrays(repl, [i0])
    [E, p_x, p_y, p_z]
    >>> P(-i0).replace_with_arrays(repl, [-i0])
    [E, -p_x, -p_y, -p_z]

    The contraction of a 1-index tensor by itself:

    >>> expr = P(i0)*P(-i0)
    >>> expr.replace_with_arrays(repl, [])
    E**2 - p_x**2 - p_y**2 - p_z**2
    """
    is_commutative = False

    def __new__(cls, name, index_types, symmetry=None, comm=0):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError('invalid name')
        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)
        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), symmetry)
        obj.comm = TensorManager.comm_symbols2i(comm)
        return obj

    @property
    def name(self):
        return self.args[0].name

    @property
    def index_types(self):
        return list(self.args[1])

    @property
    def symmetry(self):
        return self.args[2]

    @property
    def rank(self):
        return len(self.index_types)

    def __lt__(self, other):
        return (self.name, self.index_types) < (other.name, other.index_types)

    def commutes_with(self, other):
        """
        Returns ``0`` if ``self`` and ``other`` commute, ``1`` if they anticommute.

        Returns ``None`` if ``self`` and ``other`` neither commute nor anticommute.
        """
        r = TensorManager.get_comm(self.comm, other.comm)
        return r

    def _print(self):
        return '%s(%s)' % (self.name, ','.join([str(x) for x in self.index_types]))

    def __call__(self, *indices, **kw_args):
        """
        Returns a tensor with indices.

        Explanation
        ===========

        There is a special behavior in case of indices denoted by ``True``,
        they are considered auto-matrix indices, their slots are automatically
        filled, and confer to the tensor the behavior of a matrix or vector
        upon multiplication with another tensor containing auto-matrix indices
        of the same ``TensorIndexType``. This means indices get summed over the
        same way as in matrix multiplication. For matrix behavior, define two
        auto-matrix indices, for vector behavior define just one.

        Indices can also be strings, in which case the attribute
        ``index_types`` is used to convert them to proper ``TensorIndex``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, TensorHead
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b = tensor_indices('a,b', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
        >>> t = A(a, -b)
        >>> t
        A(a, -b)

        """
        updated_indices = []
        for idx, typ in zip(indices, self.index_types):
            if isinstance(idx, str):
                idx = idx.strip().replace(' ', '')
                if idx.startswith('-'):
                    updated_indices.append(TensorIndex(idx[1:], typ, is_up=False))
                else:
                    updated_indices.append(TensorIndex(idx, typ))
            else:
                updated_indices.append(idx)
        updated_indices += indices[len(updated_indices):]
        tensor = Tensor(self, updated_indices, **kw_args)
        return tensor.doit()

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError('No power on abstract tensors.')
            from .array import tensorproduct, tensorcontraction
            metrics = [_.data for _ in self.index_types]
            marray = self.data
            marraydim = marray.rank()
            for metric in metrics:
                marray = tensorproduct(marray, metric, marray)
                marray = tensorcontraction(marray, (0, marraydim), (marraydim + 1, marraydim + 2))
            return marray ** (other * S.Half)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    def _components_data_full_destroy(self):
        """
        EXPERIMENTAL: do not rely on this API method.

        Destroy components data associated to the ``TensorHead`` object, this
        checks for attached components data, and destroys components data too.
        """
        deprecate_data()
        if self.name == 'KD':
            return
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]