import numpy as np
from scipy import optimize
from statsmodels.base.model import Model
class Myfunc(NonlinearLS):

    def _predict(self, params):
        x = self.exog
        a, b, c = params
        return a * np.exp(-b * x) + c