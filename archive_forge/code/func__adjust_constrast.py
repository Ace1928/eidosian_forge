from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _adjust_constrast(self, inputs, contrast_factor):
    inp_mean = self.backend.numpy.mean(inputs, axis=-3, keepdims=True)
    inp_mean = self.backend.numpy.mean(inp_mean, axis=-2, keepdims=True)
    outputs = (inputs - inp_mean) * contrast_factor + inp_mean
    return outputs