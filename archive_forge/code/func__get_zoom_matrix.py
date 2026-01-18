from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _get_zoom_matrix(self, zooms, image_height, image_width):
    num_zooms = self.backend.shape(zooms)[0]
    x_offset = (self.backend.cast(image_width, 'float32') - 1.0) / 2.0 * (1.0 - zooms[:, 0:1])
    y_offset = (self.backend.cast(image_height, 'float32') - 1.0) / 2.0 * (1.0 - zooms[:, 1:])
    return self.backend.numpy.concatenate([zooms[:, 0:1], self.backend.numpy.zeros((num_zooms, 1)), x_offset, self.backend.numpy.zeros((num_zooms, 1)), zooms[:, 1:], y_offset, self.backend.numpy.zeros((num_zooms, 2))], axis=1)