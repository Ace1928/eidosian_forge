import _ecos
from warnings import warn
import numpy as np
from scipy import sparse
 This Python routine "unpacks" scipy sparse matrices G and A into the
        data structures that we need for calling ECOS' csolve routine.

        If G and h are both None, then we will automatically create an "empty"
        CSC matrix to use with ECOS.

        It is *not* compatible with CVXOPT spmatrix and matrix, although
        it would not be very difficult to make it compatible. We put the
        onus on the user to convert CVXOPT matrix types into numpy, scipy
        array types.
    