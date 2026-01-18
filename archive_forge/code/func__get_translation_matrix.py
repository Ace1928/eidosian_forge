from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _get_translation_matrix(self, translations):
    num_translations = self.backend.shape(translations)[0]
    return self.backend.numpy.concatenate([self.backend.numpy.ones((num_translations, 1)), self.backend.numpy.zeros((num_translations, 1)), -translations[:, 0:1], self.backend.numpy.zeros((num_translations, 1)), self.backend.numpy.ones((num_translations, 1)), -translations[:, 1:], self.backend.numpy.zeros((num_translations, 2))], axis=1)