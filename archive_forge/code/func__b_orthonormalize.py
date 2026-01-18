import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _b_orthonormalize(B, blockVectorV, blockVectorBV=None, retInvR=False):
    """B-orthonormalize the given block vector using Cholesky."""
    normalization = blockVectorV.max(axis=0) + cupy.finfo(blockVectorV.dtype).eps
    blockVectorV = blockVectorV / normalization
    if blockVectorBV is None:
        if B is not None:
            blockVectorBV = B(blockVectorV)
        else:
            blockVectorBV = blockVectorV
    else:
        blockVectorBV = blockVectorBV / normalization
    VBV = cupy.matmul(blockVectorV.T.conj(), blockVectorBV)
    try:
        VBV = _cholesky(VBV)
        VBV = linalg.inv(VBV.T)
        blockVectorV = cupy.matmul(blockVectorV, VBV)
        if B is not None:
            blockVectorBV = cupy.matmul(blockVectorBV, VBV)
        else:
            blockVectorBV = None
    except numpy.linalg.LinAlgError:
        blockVectorV = None
        blockVectorBV = None
        VBV = None
    if retInvR:
        return (blockVectorV, blockVectorBV, VBV, normalization)
    else:
        return (blockVectorV, blockVectorBV)