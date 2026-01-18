from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _store_FIM(self):
    store_dict = {}
    for i, name in enumerate(self.parameter_names):
        store_dict[name] = self.FIM[i]
    FIM_store = pd.DataFrame(store_dict)
    FIM_store.to_csv(self.store_FIM, index=False)