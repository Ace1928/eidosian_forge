import collections
import gzip
import io
import logging
import struct
import numpy as np
def _load_matrix(fin, new_format=True):
    """Load a matrix from fastText native format.

    Interprets the matrix dimensions and type from the file stream.

    Parameters
    ----------
    fin : file
        A file handle opened for reading.
    new_format : bool, optional
        True if the quant_input variable precedes
        the matrix declaration.  Should be True for newer versions of fastText.

    Returns
    -------
    :class:`numpy.array`
        The vectors as an array.
        Each vector will be a row in the array.
        The number of columns of the array will correspond to the vector size.

    """
    if _FLOAT_DTYPE is None:
        raise ValueError('bad _FLOAT_SIZE: %r' % _FLOAT_SIZE)
    if new_format:
        _struct_unpack(fin, '@?')
    num_vectors, dim = _struct_unpack(fin, '@2q')
    count = num_vectors * dim
    if isinstance(fin, gzip.GzipFile):
        logger.warning('Loading model from a compressed .gz file.  This can be slow. This is a work-around for a bug in NumPy: https://github.com/numpy/numpy/issues/13470. Consider decompressing your model file for a faster load. ')
        matrix = _fromfile(fin, _FLOAT_DTYPE, count)
    else:
        matrix = np.fromfile(fin, _FLOAT_DTYPE, count)
    assert matrix.shape == (count,), 'expected (%r,),  got %r' % (count, matrix.shape)
    matrix = matrix.reshape((num_vectors, dim))
    return matrix