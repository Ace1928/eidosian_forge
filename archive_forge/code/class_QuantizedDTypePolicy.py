from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
@keras_export(['keras.QuantizedDTypePolicy', 'keras.dtype_policies.QuantizedDTypePolicy'])
class QuantizedDTypePolicy(DTypePolicy):

    def __init__(self, name):
        super().__init__(name)
        self._quantization_mode, self._compute_dtype, self._variable_dtype = self._parse_name(name)

    def _parse_name(self, name):
        error_msg = f"Cannot convert '{name}' to a QuantizedDTypePolicy. Valid policies include 'int8_from_float32', 'int8_from_float16', 'int8_from_bfloat16', 'int8_from_mixed_float16', 'int8_from_mixed_bfloat16'."
        split_name = name.split('_from_')
        if len(split_name) != 2:
            raise ValueError(error_msg)
        mode, from_name = split_name
        if mode not in ('int8',):
            raise ValueError(error_msg)
        if from_name == 'mixed_float16':
            return (mode, 'float16', 'float32')
        elif from_name == 'mixed_bfloat16':
            return (mode, 'bfloat16', 'float32')
        try:
            dtype = backend.standardize_dtype(from_name)
            return (mode, dtype, dtype)
        except ValueError:
            raise ValueError(error_msg)

    @property
    def quantization_mode(self):
        """The quantization mode of this policy.

        Returns:
            The quantization mode of this policy, as a string.
        """
        return self._quantization_mode

    def __repr__(self):
        return f'<QuantizedDTypePolicy "{self._name}">'