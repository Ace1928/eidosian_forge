import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality
def _vi_tables(im_true, im_test, table=None, ignore_labels=()):
    """Compute probability tables used for calculating VI.

    Parameters
    ----------
    im_true, im_test : ndarray of int
        Input label images, any dimensionality.
    table : csr matrix, optional
        Pre-computed contingency table.
    ignore_labels : sequence of int, optional
        Labels to ignore when computing scores.

    Returns
    -------
    hxgy, hygx : ndarray of float
        Per-segment conditional entropies of ``im_true`` given ``im_test`` and
        vice-versa.
    """
    check_shape_equality(im_true, im_test)
    if table is None:
        pxy = contingency_table(im_true, im_test, ignore_labels=ignore_labels, normalize=True)
    else:
        pxy = table
    px = np.ravel(pxy.sum(axis=1))
    py = np.ravel(pxy.sum(axis=0))
    px_inv = sparse.diags(_invert_nonzero(px))
    py_inv = sparse.diags(_invert_nonzero(py))
    hygx = -px @ _xlogx(px_inv @ pxy).sum(axis=1)
    hxgy = -_xlogx(pxy @ py_inv).sum(axis=0) @ py
    return list(map(np.asarray, [hxgy, hygx]))