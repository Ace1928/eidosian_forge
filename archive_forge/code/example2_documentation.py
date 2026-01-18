from pyomo.environ import ConcreteModel, Var, ExternalFunction, Objective
from pyomo.opt import SolverFactory

This model is adapted from Noriyuki Yoshio's model for his and Biegler's
2021 publication in AIChE.


Yoshio, N, Biegler, L.T. Demand-based optimization of a chlorobenzene process
with high-fidelity and surrogate reactor models under trust region strategies.
AIChE J. 2021; 67:e17054. https://doi.org/10.1002/aic.17054
