from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _export_version_parser(path):
    filename = os.path.basename(path.path)
    if not (len(filename) == 10 and filename.isdigit()):
        return None
    return path._replace(export_version=int(filename))