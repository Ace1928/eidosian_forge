import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
Generate an array with block checkerboard structure for biclustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    shape : tuple of shape (n_rows, n_cols)
        The shape of the result.

    n_clusters : int or array-like or shape (n_row_clusters, n_column_clusters)
        The number of row and column clusters.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    minval : float, default=10
        Minimum value of a bicluster.

    maxval : float, default=100
        Maximum value of a bicluster.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape `shape`
        The generated array.

    rows : ndarray of shape (n_clusters, X.shape[0])
        The indicators for cluster membership of each row.

    cols : ndarray of shape (n_clusters, X.shape[1])
        The indicators for cluster membership of each column.

    See Also
    --------
    make_biclusters : Generate an array with constant block diagonal structure
        for biclustering.

    References
    ----------
    .. [1] Kluger, Y., Basri, R., Chang, J. T., & Gerstein, M. (2003).
        Spectral biclustering of microarray data: coclustering genes
        and conditions. Genome research, 13(4), 703-716.
    