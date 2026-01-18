import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestMemEventHook(_DeprecationTestCase):

    def test_mem_seteventhook(self):
        import numpy.core._multiarray_tests as ma_tests
        with pytest.warns(DeprecationWarning, match='PyDataMem_SetEventHook is deprecated'):
            ma_tests.test_pydatamem_seteventhook_start()
        a = np.zeros(1000)
        del a
        break_cycles()
        with pytest.warns(DeprecationWarning, match='PyDataMem_SetEventHook is deprecated'):
            ma_tests.test_pydatamem_seteventhook_end()