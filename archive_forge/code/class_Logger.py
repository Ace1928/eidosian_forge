import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@Deprecated(message=_LOGGER_DEPRECATION_WARNING.format(old='Logger', new='ray.tune.logger.LoggerCallback'))
@DeveloperAPI
class Logger(abc.ABC):
    """Logging interface for ray.tune.

    By default, the UnifiedLogger implementation is used which logs results in
    multiple formats (TensorBoard, rllab/viskit, plain json, custom loggers)
    at once.

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        trial: Trial object for the logger to access.
    """

    def __init__(self, config: Dict, logdir: str, trial: Optional['Trial']=None):
        self.config = config
        self.logdir = logdir
        self.trial = trial
        self._init()

    def _init(self):
        pass

    def on_result(self, result):
        """Given a result, appends it to the existing log."""
        raise NotImplementedError

    def update_config(self, config):
        """Updates the config for logger."""
        pass

    def close(self):
        """Releases all resources used by this logger."""
        pass

    def flush(self):
        """Flushes all disk writes to storage."""
        pass