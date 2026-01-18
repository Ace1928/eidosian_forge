from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def _vecmin(*s):
    """
    _vecmin(s1,s2,...) returns the componentwise minimum of s1, s2,... 
    _vecmin(s) returns the minimum component of s.
    
    The arguments can be None, scalars or 1-column dense 'd' vectors 
    with lengths equal  to 1 or equal to the maximum len(sk).  

    Returns None if one of the arguments is None.  
    """
    if not s:
        raise TypeError('_vecmin expected at least 1 argument, got 0')
    val = None
    for c in s:
        if c is None:
            return None
        elif type(c) is int or type(c) is float:
            c = matrix(c, tc='d')
        elif not _isdmatrix(c) or c.size[1] != 1:
            raise TypeError('incompatible type or size')
        if val is None:
            if len(s) == 1:
                return matrix(min(c), tc='d')
            else:
                val = +c
        elif len(val) == 1 != len(c):
            val = matrix([min(val[0], x) for x in c], tc='d')
        elif len(val) != 1 == len(c):
            val = matrix([min(c[0], x) for x in val], tc='d')
        elif len(val) == len(c):
            val = matrix([min(val[k], c[k]) for k in range(len(c))], tc='d')
        else:
            raise ValueError('incompatible dimensions')
    return val