import csv
import logging
import os
from typing import TYPE_CHECKING, Dict, TextIO
from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
def _maybe_init(self):
    """CSV outputted with Headers as first set of results."""
    if not self._initialized:
        progress_file = os.path.join(self.logdir, EXPR_PROGRESS_FILE)
        self._continuing = os.path.exists(progress_file) and os.path.getsize(progress_file) > 0
        self._file = open(progress_file, 'a')
        self._csv_out = None
        self._initialized = True