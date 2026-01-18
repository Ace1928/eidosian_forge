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
def _reset_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_x = train_df.drop([self._label_col], axis=1)
    cols = [x for x in train_x.columns if x.startswith(self._feature_prefix)]
    return (train_x[cols], train_df[self._label_col])