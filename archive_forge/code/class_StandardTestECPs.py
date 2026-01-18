import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
class StandardTestECPs:

    @staticmethod
    def test_expcone_1(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
        sth = expcone_1()
        sth.solve(solver, **kwargs)
        sth.verify_objective(places)
        sth.verify_primal_values(places)
        if duals:
            sth.check_complementarity(places)
            sth.verify_dual_values(places)
        return sth