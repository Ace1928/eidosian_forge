from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
def _get_eval_result(self, env: CallbackEnv) -> dict:
    eval_result = {}
    for entry in env.evaluation_result_list:
        data_name, eval_name, result = entry[0:3]
        if len(entry) > 4:
            stdv = entry[4]
            suffix = '-mean'
        else:
            stdv = None
            suffix = ''
        if data_name not in eval_result:
            eval_result[data_name] = {}
        eval_result[data_name][eval_name + suffix] = result
        if stdv is not None:
            eval_result[data_name][eval_name + '-stdv'] = stdv
    return eval_result