import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
@register_pass(mutates_CFG=False, analysis_only=False)
class FailPass(FunctionPass):
    _name = '_fail'

    def __init__(self, *args, **kwargs):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        raise AssertionError('unreachable')