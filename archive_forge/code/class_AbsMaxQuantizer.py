from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
@keras_export(['keras.AbsMaxQuantizer', 'keras.quantizers.AbsMaxQuantizer'])
class AbsMaxQuantizer(Quantizer):

    def __init__(self, axis, value_range=(-127, 127), epsilon=backend.epsilon(), output_dtype='int8'):
        Quantizer.__init__(self, output_dtype=output_dtype)
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = tuple(axis)
        self.value_range = value_range
        self.epsilon = epsilon

    def __call__(self, x):
        quantized_x, scale = abs_max_quantize(x, self.axis, self.value_range, self.output_dtype, self.epsilon)
        return (quantized_x, scale)

    def get_config(self):
        return {'axis': self.axis, 'value_range': self.value_range, 'epsilon': self.epsilon, 'output_dtype': self.output_dtype}