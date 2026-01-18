import os
import pytest
import numpy as np
from . import util
from numpy.f2py.crackfortran import crackfortran
class TestDataMultiplierF77(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'data_multiplier.f')]

    def test_data_stmts(self):
        assert self.module.mycom.ivar1 == 3
        assert self.module.mycom.ivar2 == 3
        assert self.module.mycom.ivar3 == 2
        assert self.module.mycom.ivar4 == 2
        assert self.module.mycom.evar5 == 0