import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
def _cases(version, filt='test%(name)s_*.mat'):
    if version == '4':
        cases = case_table4
    elif version == '5':
        cases = case_table5
    else:
        assert version == '5_rt'
        cases = case_table5_rt
    for case in cases:
        name = case['name']
        expected = case['expected']
        if filt is None:
            files = None
        else:
            use_filt = pjoin(test_data_path, filt % dict(name=name))
            files = glob(use_filt)
            assert len(files) > 0, f'No files for test {name} using filter {filt}'
        classes = case['classes']
        yield (name, files, expected, classes)