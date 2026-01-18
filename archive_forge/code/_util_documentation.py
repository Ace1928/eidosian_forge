import math
import cupy
Helper function for SciPy argument validation.

    Many CuPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array-like
        The array-like input
    check_finite : bool, optional
        By default True. To check whether the input matrices contain
        only finite numbers. Disabling may give a performance gain,
        but may result in problems (crashes, non-termination) if the
        inputs do contain infinites or NaNs
    sparse_ok : bool, optional
        By default False. True if cupy sparse matrices are allowed
    objects_ok : bool, optional
        By default False. True if arrays with dype('O') are allowed
    mask_ok : bool, optional
        By default False. True if masked arrays are allowed.
    as_inexact : bool, optional
        By default False. True to convert the input array to a
        cupy.inexact dtype

    Returns
    -------
    ret : cupy.ndarray
        The converted validated array

    