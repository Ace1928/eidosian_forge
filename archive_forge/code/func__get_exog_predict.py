import numpy as np
from scipy import stats
import pandas as pd
def _get_exog_predict(self, exog=None, transform=True, row_labels=None):
    """Prepare or transform exog for prediction

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    exog : ndarray
        Prediction exog
    row_labels : list of str
        Labels or pandas index for rows of prediction
    """
    if transform and hasattr(self.model, 'formula') and (exog is not None):
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)
    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)
    else:
        exog = self.model.exog
        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    return (exog, row_labels)