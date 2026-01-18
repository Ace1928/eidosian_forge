from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
class DistributedResults(LikelihoodModelResults):
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        Class instance for model used for distributed data,
        this particular instance uses fake data and is really
        only to allow use of methods like predict.
    params : ndarray
        Parameter estimates from the fit model.
    """

    def __init__(self, model, params):
        super().__init__(model, params)

    def predict(self, exog, *args, **kwargs):
        """Calls self.model.predict for the provided exog.  See
        Results.predict.

        Parameters
        ----------
        exog : array_like NOT optional
            The values for which we want to predict, unlike standard
            predict this is NOT optional since the data in self.model
            is fake.
        *args :
            Some models can take additional arguments. See the
            predict method of the model for the details.
        **kwargs :
            Some models can take additional keywords arguments. See the
            predict method of the model for the details.

        Returns
        -------
            prediction : ndarray, pandas.Series or pandas.DataFrame
            See self.model.predict
        """
        return self.model.predict(self.params, exog, *args, **kwargs)