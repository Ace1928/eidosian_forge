import csv
import io
import math
from tensorboard.plugins.hparams import error
def _get_metric_id(metric):
    return metric.group + '.' + metric.tag