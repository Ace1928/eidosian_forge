import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def _count_callbacks(callbacks: Optional[List['Callback']]) -> Dict[str, int]:
    """Creates a map of callback class name -> count given a list of callbacks."""
    from ray.tune import Callback
    from ray.tune.logger import LoggerCallback
    from ray.tune.utils.callback import DEFAULT_CALLBACK_CLASSES
    from ray.air.integrations.wandb import WandbLoggerCallback
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from ray.air.integrations.comet import CometLoggerCallback
    from ray.tune.logger.aim import AimLoggerCallback
    built_in_callbacks = (WandbLoggerCallback, MLflowLoggerCallback, CometLoggerCallback, AimLoggerCallback) + DEFAULT_CALLBACK_CLASSES
    callback_names = [callback_cls.__name__ for callback_cls in built_in_callbacks]
    callback_counts = collections.defaultdict(int)
    callbacks = callbacks or []
    for callback in callbacks:
        if not isinstance(callback, Callback):
            continue
        callback_name = callback.__class__.__name__
        if callback_name in callback_names:
            callback_counts[callback_name] += 1
        elif isinstance(callback, LoggerCallback):
            callback_counts['CustomLoggerCallback'] += 1
        else:
            callback_counts['CustomCallback'] += 1
    return callback_counts