import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
from pyomo.common.dependencies import pandas as pd

This example uses an iteration callback with a functor to store
values from each iteration in a class
