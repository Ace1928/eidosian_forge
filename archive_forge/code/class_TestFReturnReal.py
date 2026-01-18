import platform
import pytest
import numpy as np
from numpy import array
from . import util
class TestFReturnReal(TestReturnReal):
    sources = [util.getpath('tests', 'src', 'return_real', 'foo77.f'), util.getpath('tests', 'src', 'return_real', 'foo90.f90')]

    @pytest.mark.parametrize('name', 't0,t4,t8,td,s0,s4,s8,sd'.split(','))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize('name', 't0,t4,t8,td,s0,s4,s8,sd'.split(','))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_real, name), name)