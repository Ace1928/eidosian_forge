from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
def _dbscale(self, inputs):
    log_spec = 10.0 * self.backend.numpy.log10(self.backend.numpy.maximum(inputs, self.min_power))
    ref_value = self.backend.numpy.abs(self.backend.convert_to_tensor(self.ref_power))
    log_spec -= 10.0 * self.backend.numpy.log10(self.backend.numpy.maximum(ref_value, self.min_power))
    log_spec = self.backend.numpy.maximum(log_spec, self.backend.numpy.max(log_spec) - self.top_db)
    return log_spec