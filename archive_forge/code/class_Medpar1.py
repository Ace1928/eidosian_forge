import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids
class Medpar1:
    """
    The medpar1 data can be found here.

    https://www.stata-press.com/data/hh2/medpar1
    """

    def __init__(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stata_medpar1_glm.csv')
        data = pd.read_csv(filename).to_records()
        self.endog = data.los
        dummies = pd.get_dummies(data.admitype, prefix='race', drop_first=True, dtype=float)
        design = np.column_stack((data.codes, dummies)).astype(float)
        self.exog = add_constant(design, prepend=False)