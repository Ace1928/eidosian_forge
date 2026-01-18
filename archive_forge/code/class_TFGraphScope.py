import sys
from keras.src import backend as backend_module
from keras.src.backend.common import global_state
class TFGraphScope:

    def __init__(self):
        self._original_value = global_state.get_global_attribute('in_tf_graph_scope', False)

    def __enter__(self):
        global_state.set_global_attribute('in_tf_graph_scope', True)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute('in_tf_graph_scope', self._original_value)