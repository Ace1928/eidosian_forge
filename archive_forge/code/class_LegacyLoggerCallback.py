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
@DeveloperAPI
class LegacyLoggerCallback(LoggerCallback):
    """Supports logging to trial-specific `Logger` classes.

    Previously, Ray Tune logging was handled via `Logger` classes that have
    been instantiated per-trial. This callback is a fallback to these
    `Logger`-classes, instantiating each `Logger` class for each trial
    and logging to them.

    Args:
        logger_classes: Logger classes that should
            be instantiated for each trial.

    """

    def __init__(self, logger_classes: Iterable[Type[Logger]]):
        self.logger_classes = list(logger_classes)
        self._class_trial_loggers: Dict[Type[Logger], Dict['Trial', Logger]] = {}

    def log_trial_start(self, trial: 'Trial'):
        trial.init_local_path()
        for logger_class in self.logger_classes:
            trial_loggers = self._class_trial_loggers.get(logger_class, {})
            if trial not in trial_loggers:
                logger = logger_class(trial.config, trial.local_path, trial)
                trial_loggers[trial] = logger
            self._class_trial_loggers[logger_class] = trial_loggers

    def log_trial_restore(self, trial: 'Trial'):
        for logger_class, trial_loggers in self._class_trial_loggers.items():
            if trial in trial_loggers:
                trial_loggers[trial].flush()

    def log_trial_save(self, trial: 'Trial'):
        for logger_class, trial_loggers in self._class_trial_loggers.items():
            if trial in trial_loggers:
                trial_loggers[trial].flush()

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        for logger_class, trial_loggers in self._class_trial_loggers.items():
            if trial in trial_loggers:
                trial_loggers[trial].on_result(result)

    def log_trial_end(self, trial: 'Trial', failed: bool=False):
        for logger_class, trial_loggers in self._class_trial_loggers.items():
            if trial in trial_loggers:
                trial_loggers[trial].close()