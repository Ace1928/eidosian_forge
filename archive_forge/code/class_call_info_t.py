import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
class call_info_t(ct.Structure):
    _fields_ = [('strided_loop', strided_loop_t), ('context', ct.c_void_p), ('auxdata', ct.c_void_p), ('requires_pyapi', ct.c_byte), ('no_floatingpoint_errors', ct.c_byte)]