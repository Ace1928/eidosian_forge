import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
def _named_tempfile_func(error_class):
    """Create a mock function for NamedTemporaryFile that always raises.

    Parameters
    ----------
    error_class : exception class
        The error that should be raised when asking for a NamedTemporaryFile.

    Returns
    -------
    named_temp_file : callable
        A function that always raises the desired error.

    Notes
    -----
    Although this function has general utility for raising errors, it is
    expected to be used to raise errors that ``tempfile.NamedTemporaryFile``
    from the Python standard library could raise. As of this writing, these
    are ``FileNotFoundError``, ``FileExistsError``, ``PermissionError``, and
    ``BaseException``. See
    `this comment <https://github.com/scikit-image/scikit-image/issues/3785#issuecomment-486598307>`__  # noqa
    for more information.
    """

    def named_temp_file(*args, **kwargs):
        raise error_class()
    return named_temp_file