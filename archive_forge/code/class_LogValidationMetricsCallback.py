import logging
import math
import time
from .model import save_checkpoint
class LogValidationMetricsCallback(object):
    """Just logs the eval metrics at the end of an epoch."""

    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)