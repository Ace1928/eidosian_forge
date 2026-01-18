import cupy
Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Args:
        x (array_like): Matrix of M vectors in K dimensions.
        y (array_like): Matrix of N vectors in K dimensions.
        p (float): Which Minkowski p-norm to use (1 <= p <= infinity).
            Default=2.0
    Returns:
        result (cupy.ndarray): Matrix containing the distance from every
            vector in `x` to every vector in `y`, (size M, N).
    