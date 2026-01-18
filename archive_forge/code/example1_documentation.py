from pyomo.environ import (
from pyomo.opt import SolverFactory

This model is an adaptation of Eason & Biegler's example in the
original version of the Trust Region solver.

Eason, J.P. and Biegler, L.T. (2018), Advanced trust region optimization
strategies for glass box/black box models. AIChE J, 64: 3934-3943.
https://doi.org/10.1002/aic.16364
