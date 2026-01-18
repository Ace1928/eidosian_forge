from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def _eigenvects(M, error_when_incomplete=True, iszerofunc=_iszero, *, chop=False, **flags):
    """Compute eigenvectors of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        Raise an error when not all eigenvalues are computed. This is
        caused by ``roots`` not returning a full list of eigenvalues.

    iszerofunc : function, optional
        Specifies a zero testing function to be used in ``rref``.

        Default value is ``_iszero``, which uses SymPy's naive and fast
        default assumption handler.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        is tested as non-zero, and ``None`` if it is undecidable.

    simplify : bool or function, optional
        If ``True``, ``as_content_primitive()`` will be used to tidy up
        normalization artifacts.

        It will also be used by the ``nullspace`` routine.

    chop : bool or positive number, optional
        If the matrix contains any Floats, they will be changed to Rationals
        for computation purposes, but the answers will be returned after
        being evaluated with evalf. The ``chop`` flag is passed to ``evalf``.
        When ``chop=True`` a default precision will be used; a number will
        be interpreted as the desired level of precision.

    Returns
    =======

    ret : [(eigenval, multiplicity, eigenspace), ...]
        A ragged list containing tuples of data obtained by ``eigenvals``
        and ``nullspace``.

        ``eigenspace`` is a list containing the ``eigenvector`` for each
        eigenvalue.

        ``eigenvector`` is a vector in the form of a ``Matrix``. e.g.
        a vector of length 3 is returned as ``Matrix([a_1, a_2, a_3])``.

    Raises
    ======

    NotImplementedError
        If failed to compute nullspace.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]

    See Also
    ========

    eigenvals
    MatrixSubspaces.nullspace
    """
    simplify = flags.get('simplify', True)
    primitive = flags.get('simplify', False)
    flags.pop('simplify', None)
    flags.pop('multiple', None)
    if not isinstance(simplify, FunctionType):
        simpfunc = _simplify if simplify else lambda x: x
    has_floats = M.has(Float)
    if has_floats:
        if all((x.is_number for x in M)):
            return _eigenvects_mpmath(M)
        from sympy.simplify import nsimplify
        M = M.applyfunc(lambda x: nsimplify(x, rational=True))
    ret = _eigenvects_DOM(M)
    if ret is None:
        ret = _eigenvects_sympy(M, iszerofunc, simplify=simplify, **flags)
    if primitive:

        def denom_clean(l):
            return [(v / gcd(list(v))).applyfunc(simpfunc) for v in l]
        ret = [(val, mult, denom_clean(es)) for val, mult, es in ret]
    if has_floats:
        ret = [(val.evalf(chop=chop), mult, [v.evalf(chop=chop) for v in es]) for val, mult, es in ret]
    return ret