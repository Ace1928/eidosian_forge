import numpy as np
import pandas as pd
from statsmodels.graphics.utils import maybe_name_or_idx
def _guess_endog_name(self, model, typ):
    if hasattr(model, 'formula'):
        return model.formula.split('~')[0].strip()
    else:
        raise ValueError('cannot infer %s name without formula' % typ)