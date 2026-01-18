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
def _export_eval_result(self, eval_result, is_the_final_export):
    """Export `eval_result` according to exporters in `EvalSpec`."""
    export_dir_base = os.path.join(tf.compat.as_str_any(self._estimator.model_dir), tf.compat.as_str_any('export'))
    export_results = []
    for exporter in self._eval_spec.exporters:
        export_results.append(exporter.export(estimator=self._estimator, export_path=os.path.join(tf.compat.as_str_any(export_dir_base), tf.compat.as_str_any(exporter.name)), checkpoint_path=eval_result.checkpoint_path, eval_result=eval_result.metrics, is_the_final_export=is_the_final_export))
    return export_results