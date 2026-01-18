from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
@keras_export(['keras.quantizers.abs_max_quantize'])
def abs_max_quantize(inputs, axis, value_range=(-127, 127), dtype='int8', epsilon=backend.epsilon()):
    scale = ops.divide(value_range[1], ops.add(ops.max(ops.abs(inputs), axis=axis, keepdims=True), epsilon))
    outputs = ops.multiply(inputs, scale)
    outputs = ops.clip(ops.round(outputs), value_range[0], value_range[1])
    outputs = ops.cast(outputs, dtype)
    return (outputs, scale)