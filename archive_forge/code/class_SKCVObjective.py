import os
import cloudpickle
from tune.api.factory import TUNE_OBJECT_FACTORY
from typing import Any, Optional, Tuple
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from triad import FileSystem
from tune import NonIterativeObjectiveFunc, Trial, TrialReport
from tune.constants import (
from tune_sklearn.utils import to_sk_model, to_sk_model_expr
class SKCVObjective(SKObjective):

    def __init__(self, scoring: Any, cv: int=5, feature_prefix: str='', label_col: str='label', checkpoint_path: Optional[str]=None) -> None:
        super().__init__(scoring=scoring, feature_prefix=feature_prefix, label_col=label_col, checkpoint_path=checkpoint_path)
        self._cv = cv

    def run(self, trial: Trial) -> TrialReport:
        params = dict(trial.params.simple_value)
        if trial.trial_id != self._last_id:
            self._model_type = to_sk_model(params.pop(SPACE_MODEL_NAME))
            self._model_expr = to_sk_model_expr(self._model_type)
            self._train_x, self._train_y = self._reset_xy(trial.dfs[TUNE_DATASET_DF_DEFAULT_NAME])
            self._last_id = trial.trial_id
        else:
            params.pop(SPACE_MODEL_NAME)
        model = self._model_type(**params)
        s = cross_val_score(model, self._train_x, self._train_y, cv=self._cv, scoring=self._scoring)
        metadata = dict(model=self._model_expr, cv_scores=[float(x) for x in s])
        if self._checkpoint_path is not None:
            model.fit(self._train_x, self._train_y)
            fp = os.path.join(self._checkpoint_path, str(uuid4()) + '.pkl')
            with FileSystem().openbin(fp, mode='wb') as f:
                cloudpickle.dump(model, f)
            metadata['checkpoint_path'] = fp
        metric = float(np.mean(s))
        return TrialReport(trial, metric=metric, metadata=metadata, sort_metric=self.generate_sort_metric(metric))