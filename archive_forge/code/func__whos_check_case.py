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
def _whos_check_case(name, files, case, classes):
    for file_name in files:
        label = f'test {name}; file {file_name}'
        whos = whosmat(file_name)
        expected_whos = [(k, expected.shape, classes[k]) for k, expected in case.items()]
        whos.sort()
        expected_whos.sort()
        assert_equal(whos, expected_whos, f'{label}: {whos!r} != {expected_whos!r}')