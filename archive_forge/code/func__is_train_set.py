from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def _is_train_set(self, ds_name: str, eval_name: str, env: CallbackEnv) -> bool:
    """Check, by name, if a given Dataset is the training data."""
    if ds_name == 'cv_agg' and eval_name == 'train':
        return True
    if isinstance(env.model, Booster) and ds_name == env.model._train_data_name:
        return True
    return False