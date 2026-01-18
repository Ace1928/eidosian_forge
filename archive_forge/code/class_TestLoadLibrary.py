import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
@pytest.mark.skipif(ctypes is None, reason='ctypes not available in this python')
@pytest.mark.skipif(sys.platform == 'cygwin', reason='Known to fail on cygwin')
class TestLoadLibrary:

    def test_basic(self):
        loader_path = np.core._multiarray_umath.__file__
        out1 = load_library('_multiarray_umath', loader_path)
        out2 = load_library(Path('_multiarray_umath'), loader_path)
        out3 = load_library('_multiarray_umath', Path(loader_path))
        out4 = load_library(b'_multiarray_umath', loader_path)
        assert isinstance(out1, ctypes.CDLL)
        assert out1 is out2 is out3 is out4

    def test_basic2(self):
        try:
            so_ext = sysconfig.get_config_var('EXT_SUFFIX')
            load_library('_multiarray_umath%s' % so_ext, np.core._multiarray_umath.__file__)
        except ImportError as e:
            msg = 'ctypes is not available on this python: skipping the test (import error was: %s)' % str(e)
            print(msg)