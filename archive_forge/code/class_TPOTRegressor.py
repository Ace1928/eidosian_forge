from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing
import numpy as np
from .base import TPOTBase
from .config.classifier import classifier_config_dict
from .config.regressor import regressor_config_dict
class TPOTRegressor(TPOTBase):
    """TPOT estimator for regression problems."""
    scoring_function = 'neg_mean_squared_error'
    default_config_dict = regressor_config_dict
    classification = False
    regression = True

    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work with the passed data set.

        """
        self.pretest_X, _, self.pretest_y, _ = train_test_split(features, target, random_state=self.random_state, test_size=None, train_size=min(50, int(0.9 * features.shape[0])))