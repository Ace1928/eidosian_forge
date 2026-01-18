from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
@keras_export(['keras.FloatDTypePolicy', 'keras.dtype_policies.FloatDTypePolicy'])
class FloatDTypePolicy(DTypePolicy):

    def __init__(self, name):
        super().__init__(name)
        self._compute_dtype, self._variable_dtype = self._parse_name(name)

    def _parse_name(self, name):
        if name == 'mixed_float16':
            return ('float16', 'float32')
        elif name == 'mixed_bfloat16':
            return ('bfloat16', 'float32')
        try:
            dtype = backend.standardize_dtype(name)
            return (dtype, dtype)
        except ValueError:
            raise ValueError(f"Cannot convert '{name}' to a mixed precision FloatDTypePolicy. Valid policies include 'mixed_float16', 'mixed_bfloat16', and the name of any float dtype such as 'float32'.")

    def __repr__(self):
        return f'<FloatDTypePolicy "{self._name}">'