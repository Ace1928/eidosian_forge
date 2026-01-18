from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing
import numpy as np
from .base import TPOTBase
from .config.classifier import classifier_config_dict
from .config.regressor import regressor_config_dict
class TPOTClassifier(TPOTBase):
    """TPOT estimator for classification problems."""
    scoring_function = 'accuracy'
    default_config_dict = classifier_config_dict
    classification = True
    regression = False

    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work
        with the passed data set.

        This is not intend for anything other than perfunctory dataset
        pipeline compatibility testing
        """
        num_unique_target = len(np.unique(target))
        train_size = max(min(50, int(0.9 * features.shape[0])), num_unique_target)
        self.pretest_X, _, self.pretest_y, _ = train_test_split(features, target, random_state=self.random_state, test_size=None, train_size=train_size)
        if not np.array_equal(np.unique(target), np.unique(self.pretest_y)):
            unique_target_idx = np.unique(target, return_index=True)[1]
            self.pretest_y[0:unique_target_idx.shape[0]] = _safe_indexing(target, unique_target_idx)