import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.uint64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        