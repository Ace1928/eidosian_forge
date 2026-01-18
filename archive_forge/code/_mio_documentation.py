from contextlib import contextmanager
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer

    List variables inside a MATLAB file.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    variables : list of tuples
        A list of tuples, where each tuple holds the matrix name (a string),
        its shape (tuple of ints), and its data class (a string).
        Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
        int64, uint64, single, double, cell, struct, object, char, sparse,
        function, opaque, logical, unknown.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 python library to read matlab 7.3 format mat
    files (e.g. h5py). Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    .. versionadded:: 0.12.0

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.io import savemat, whosmat

    Create some arrays, and use `savemat` to write them to a ``BytesIO``
    instance.

    >>> a = np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32)
    >>> b = np.geomspace(1, 10, 5)
    >>> f = BytesIO()
    >>> savemat(f, {'a': a, 'b': b})

    Use `whosmat` to inspect ``f``.  Each tuple in the output list gives
    the name, shape and data type of the array in ``f``.

    >>> whosmat(f)
    [('a', (2, 3), 'int32'), ('b', (1, 5), 'double')]

    