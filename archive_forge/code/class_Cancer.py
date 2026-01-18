import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids
class Cancer:
    """
    The Cancer data can be found here

    https://www.stata-press.com/data/r10/rmain.html
    """

    def __init__(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stata_cancer_glm.csv')
        data = pd.read_csv(filename)
        self.endog = data.studytime
        dummies = pd.get_dummies(pd.Series(data.drug, dtype='category'), drop_first=True)
        design = np.column_stack((data.age, dummies)).astype(float)
        self.exog = add_constant(design, prepend=False)