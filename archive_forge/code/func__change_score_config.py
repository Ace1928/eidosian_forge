import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def _change_score_config(self, config):
    if config is not None:
        if isinstance(config, ScoreType):
            if config == ScoreType.Abs:
                config = ScoreConfig.abs_score()
            elif config == ScoreType.Rel:
                config = ScoreConfig.rel_score()
            else:
                raise CatBoostError('Unknown scoreType {}'.format(config))
        if self._score_config != config:
            self._score_config = config
            self.__clear_comparisons()