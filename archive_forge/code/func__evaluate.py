from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _evaluate(self, train_session):
    var_name_to_value = train_session.run(self._var_name_to_train_var)
    placeholder_to_value = {self._var_name_to_placeholder[v_name]: var_name_to_value[v_name] for v_name in var_name_to_value}

    def feed_variables(scaffold, session):
        del scaffold
        session.run(self._var_feed_op, feed_dict=placeholder_to_value)
    scaffold = tf.compat.v1.train.Scaffold(init_fn=feed_variables, copy_from_scaffold=self._scaffold)
    with self._graph.as_default():
        self._estimator._evaluate_run(checkpoint_path=None, scaffold=scaffold, update_op=self._update_op, eval_dict=self._eval_dict, all_hooks=self._all_hooks, output_dir=self._eval_dir)
    self._timer.update_last_triggered_step(self._iter_count)