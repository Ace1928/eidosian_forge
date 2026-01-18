from sympy.utilities.iterables import \
from .common import NonSquareMatrixError
def _strongly_connected_components_decomposition(M, lower=True):
    """Decomposes a square matrix into block triangular form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Parameters
    ==========

    lower : bool
        Makes $B$ lower block triangular when ``True``.
        Otherwise, makes $B$ upper block triangular.

    Returns
    =======

    P, B : PermutationMatrix, BlockMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block triangular matrix of
        the result of the permutation.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])

    A lower block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((8)(1 4 3 2 6)(5 7))
    >>> pprint(B)
    [[44  43  45]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[34  33  35]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[54  53  55]   [0  0  0]     [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [62  61  60]  [66  68  67]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [82  81  80]  [86  88  87]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [72  71  70]  [76  78  77]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True

    An upper block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition(lower=False)
    >>> pprint(P)
    PermutationMatrix((0 1 5 7 4 3 2 8 6))
    >>> pprint(B)
    [[66  68  67]  [62  61  60]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[86  88  87]  [82  81  80]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[76  78  77]  [72  71  70]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]     [0  0  0]    [44  43  45]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [34  33  35]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [54  53  55]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix
    iblocks = M.strongly_connected_components()
    if not lower:
        iblocks = list(reversed(iblocks))
    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)
    rows = []
    for a in iblocks:
        cols = []
        for b in iblocks:
            cols.append(M[a, b])
        rows.append(cols)
    B = BlockMatrix(rows)
    return (P, B)