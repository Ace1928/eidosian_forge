from pyomo.common.dependencies import pandas as pd
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
def SSE_multisensor(model, data):
    expr = (float(data.iloc[0]['ca1']) - model.ca) ** 2 * (1 / 3) + (float(data.iloc[0]['ca2']) - model.ca) ** 2 * (1 / 3) + (float(data.iloc[0]['ca3']) - model.ca) ** 2 * (1 / 3) + (float(data.iloc[0]['cb']) - model.cb) ** 2 + (float(data.iloc[0]['cc1']) - model.cc) ** 2 * (1 / 2) + (float(data.iloc[0]['cc2']) - model.cc) ** 2 * (1 / 2) + (float(data.iloc[0]['cd']) - model.cd) ** 2
    return expr