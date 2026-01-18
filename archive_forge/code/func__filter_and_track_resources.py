import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _filter_and_track_resources(self):
    """Track resources used by endpoints / referenced in `track()` calls."""
    fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
    tvs, ntvs = _list_variables_used_by_fns(fns)
    self._all_variables = list(tvs + ntvs)
    self._misc_assets = []
    from keras.src.layers.preprocessing.index_lookup import IndexLookup
    if hasattr(self, '_tracked'):
        for root in self._tracked:
            descendants = tf.train.TrackableView(root).descendants()
            for trackable in descendants:
                if isinstance(trackable, IndexLookup):
                    self._misc_assets.append(trackable)