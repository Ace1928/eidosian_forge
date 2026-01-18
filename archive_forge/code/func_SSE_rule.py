from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
def SSE_rule(m):
    return sum(((data.y[i] - m.response_function[data.hour[i]]) ** 2 for i in data.index))