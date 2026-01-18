import osqp
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol

        Setup equality constrained feasibility problem

            min     0
            st      A x = l = u
        