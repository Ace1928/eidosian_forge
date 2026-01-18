import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
def _variable_pos(self, var, model):
    if model == 'mediator':
        mod = self.mediator_model
    else:
        mod = self.outcome_model
    if var == 'mediator':
        return maybe_name_or_idx(self.mediator, mod)[1]
    exp = self.exposure
    exp_is_2 = len(exp) == 2 and (not isinstance(exp, str))
    if exp_is_2:
        if model == 'outcome':
            return exp[0]
        elif model == 'mediator':
            return exp[1]
    else:
        return maybe_name_or_idx(exp, mod)[1]