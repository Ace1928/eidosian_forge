from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _gather_state(self, features):
    """Returns `features` with state packed, indicates if packing was done."""
    prefixed_state_re = re.compile('^' + feature_keys.State.STATE_PREFIX + '_(\\d+)$')
    numbered_state = []
    for key, tensor in features.items():
        search_result = prefixed_state_re.search(key)
        if search_result:
            numbered_state.append((int(search_result.group(1)), key, tensor))
    if not numbered_state:
        return (features, False)
    features = features.copy()
    for _, key, _ in numbered_state:
        del features[key]
    numbered_state.sort(key=lambda number, *_: number)
    features[feature_keys.State.STATE_TUPLE] = tf.nest.pack_sequence_as(structure=self.model.get_start_state(), flat_sequence=[tensor for _, _, tensor in numbered_state])
    return (features, True)