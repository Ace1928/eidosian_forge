import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class Test_AdaptiveStepsize:

    def setup_method(self):
        self.stepsize = 1.0
        self.ts = RandomDisplacement(stepsize=self.stepsize)
        self.target_accept_rate = 0.5
        self.takestep = AdaptiveStepsize(takestep=self.ts, verbose=False, accept_rate=self.target_accept_rate)

    def test_adaptive_increase(self):
        x = 0.0
        self.takestep(x)
        self.takestep.report(False)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_adaptive_decrease(self):
        x = 0.0
        self.takestep(x)
        self.takestep.report(True)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)

    def test_all_accepted(self):
        x = 0.0
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_all_rejected(self):
        x = 0.0
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)