import warnings
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
def _get_embeddings_with_merged_lora(self):
    if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
        embeddings_value = self._embeddings
        embeddings_scale = self.embeddings_scale
        if self.lora_enabled:
            embeddings_value = ops.divide(embeddings_value, embeddings_scale)
            embeddings_value = ops.add(embeddings_value, ops.matmul(self.lora_embeddings_a, self.lora_embeddings_b))
            embeddings_value, embeddings_scale = quantizers.abs_max_quantize(embeddings_value, axis=0)
            embeddings_scale = ops.squeeze(embeddings_scale, axis=0)
        return (embeddings_value, embeddings_scale)
    return (self.embeddings, None)