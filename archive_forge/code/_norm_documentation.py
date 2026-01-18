import numpy
import cupy
import cupyx.scipy.sparse
Norm of a cupy.scipy.spmatrix

    This function is able to return one of seven different sparse matrix norms,
    depending on the value of the ``ord`` parameter.

    Args:
        x (sparse matrix) : Input sparse matrix.
        ord (non-zero int, inf, -inf, 'fro', optional) : Order of the norm (see
            table under ``Notes``). inf means numpy's `inf` object.
        axis : (int, 2-tuple of ints, None, optional): If `axis` is an
            integer, it specifies the axis of `x` along which to
            compute the vector norms.  If `axis` is a 2-tuple, it specifies the
            axes that hold 2-D matrices, and the matrix norms of these matrices
            are computed.  If `axis` is None then either a vector norm
            (when `x` is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    Returns:
        ndarray : 0-D or 1-D array or norm(s).

    .. seealso:: :func:`scipy.sparse.linalg.norm`
    