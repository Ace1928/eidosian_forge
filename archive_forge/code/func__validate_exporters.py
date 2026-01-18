from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _validate_exporters(exporters):
    """Validates `exporters` and returns them as a tuple."""
    if not exporters:
        return ()
    if isinstance(exporters, exporter_lib.Exporter):
        exporters = [exporters]
    unique_names = []
    try:
        for exporter in exporters:
            if not isinstance(exporter, exporter_lib.Exporter):
                raise TypeError
            if not exporter.name:
                full_list_of_names = [e.name for e in exporters]
                raise ValueError('An Exporter cannot have a name that is `None` or empty. All exporter names: {}'.format(full_list_of_names))
            if not isinstance(exporter.name, six.string_types):
                raise ValueError('An Exporter must have a string name. Given: {}'.format(type(exporter.name)))
            if exporter.name in unique_names:
                full_list_of_names = [e.name for e in exporters]
                raise ValueError('`exporters` must have unique names. Such a name cannot be `None`. All exporter names: {}'.format(full_list_of_names))
            unique_names.append(exporter.name)
    except TypeError:
        raise TypeError('`exporters` must be an Exporter, an iterable of Exporter, or `None`, found %s.' % exporters)
    return tuple(exporters)