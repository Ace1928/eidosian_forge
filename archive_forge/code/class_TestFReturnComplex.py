import pytest
from numpy import array
from . import util
class TestFReturnComplex(TestReturnComplex):
    sources = [util.getpath('tests', 'src', 'return_complex', 'foo77.f'), util.getpath('tests', 'src', 'return_complex', 'foo90.f90')]

    @pytest.mark.parametrize('name', 't0,t8,t16,td,s0,s8,s16,sd'.split(','))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize('name', 't0,t8,t16,td,s0,s8,s16,sd'.split(','))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_complex, name), name)