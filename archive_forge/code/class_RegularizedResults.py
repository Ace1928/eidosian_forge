import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class RegularizedResults(Results):
    """
    Results for models estimated using regularization

    Parameters
    ----------
    model : Model
        The model instance used to estimate the parameters.
    params : ndarray
        The estimated (regularized) parameters.
    """

    def __init__(self, model, params):
        super().__init__(model, params)

    @cache_readonly
    def fittedvalues(self):
        """
        The predicted values from the model at the estimated parameters.
        """
        return self.model.predict(self.params)