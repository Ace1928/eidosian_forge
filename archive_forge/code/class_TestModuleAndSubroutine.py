import os
import pytest
import numpy as np
from . import util
class TestModuleAndSubroutine(util.F2PyTest):
    module_name = 'example'
    sources = [util.getpath('tests', 'src', 'regression', 'gh25337', 'data.f90'), util.getpath('tests', 'src', 'regression', 'gh25337', 'use_data.f90')]

    @pytest.mark.slow
    def test_gh25337(self):
        self.module.data.set_shift(3)
        assert 'data' in dir(self.module)