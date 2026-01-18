import json
import logging
import numpy as np
import os
from typing import TYPE_CHECKING, Dict, TextIO
from ray.air.constants import (
import ray.cloudpickle as cloudpickle
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.utils.util import SafeFallbackEncoder
from ray.util.annotations import Deprecated, PublicAPI
@Deprecated(message=_LOGGER_DEPRECATION_WARNING.format(old='JsonLogger', new='ray.tune.json.JsonLoggerCallback'), warning=True)
@PublicAPI
class JsonLogger(Logger):
    """Logs trial results in json format.

    Also writes to a results file and param.json file when results or
    configurations are updated. Experiments must be executed with the
    JsonLogger to be compatible with the ExperimentAnalysis tool.
    """

    def _init(self):
        self.update_config(self.config)
        local_file = os.path.join(self.logdir, EXPR_RESULT_FILE)
        self.local_out = open(local_file, 'a')

    def on_result(self, result: Dict):
        json.dump(result, self, cls=SafeFallbackEncoder)
        self.write('\n')
        self.local_out.flush()

    def write(self, b):
        self.local_out.write(b)

    def flush(self):
        if not self.local_out.closed:
            self.local_out.flush()

    def close(self):
        self.local_out.close()

    def update_config(self, config: Dict):
        self.config = config
        config_out = os.path.join(self.logdir, EXPR_PARAM_FILE)
        with open(config_out, 'w') as f:
            json.dump(self.config, f, indent=2, sort_keys=True, cls=SafeFallbackEncoder)
        config_pkl = os.path.join(self.logdir, EXPR_PARAM_PICKLE_FILE)
        with open(config_pkl, 'wb') as f:
            cloudpickle.dump(self.config, f)