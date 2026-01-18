import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
def EnforceCattleFeedRequirement_rule(m, i):
    return farmer.CattleFeedRequirement[i] <= farmer.crop_yield[scenario][i] * m.devoted_acreage[i] + m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i]