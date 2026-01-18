from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
def _melscale(self, inputs):
    matrix = self.linear_to_mel_weight_matrix(num_mel_bins=self.num_mel_bins, num_spectrogram_bins=self.backend.shape(inputs)[-1], sampling_rate=self.sampling_rate, lower_edge_hertz=self.min_freq, upper_edge_hertz=self.max_freq)
    return self.backend.numpy.tensordot(inputs, matrix, axes=1)