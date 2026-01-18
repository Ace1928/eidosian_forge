import numpy as np

        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A vector containing the gradient of the penalty
        with respect to each element in the lower triangle
        of `mat`.
        