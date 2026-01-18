from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _effects_at(effects, at):
    if at == 'all':
        effects = effects
    elif at == 'overall':
        effects = effects.mean(0)
    else:
        effects = effects[0, :]
    return effects