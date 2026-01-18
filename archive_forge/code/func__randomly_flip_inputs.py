from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _randomly_flip_inputs(self, inputs):
    inputs_shape = self.backend.shape(inputs)
    unbatched = len(inputs_shape) == 3
    if unbatched:
        inputs = self.backend.numpy.expand_dims(inputs, axis=0)
        inputs_shape = self.backend.shape(inputs)
    batch_size = inputs_shape[0]
    flipped_outputs = inputs
    seed_generator = self._get_seed_generator(self.backend._backend)
    if self.mode == HORIZONTAL or self.mode == HORIZONTAL_AND_VERTICAL:
        flipped_outputs = self.backend.numpy.where(self.backend.random.uniform(shape=(batch_size, 1, 1, 1), seed=seed_generator) <= 0.5, self.backend.numpy.flip(flipped_outputs, axis=-2), flipped_outputs)
    if self.mode == VERTICAL or self.mode == HORIZONTAL_AND_VERTICAL:
        flipped_outputs = self.backend.numpy.where(self.backend.random.uniform(shape=(batch_size, 1, 1, 1), seed=seed_generator) <= 0.5, self.backend.numpy.flip(flipped_outputs, axis=-3), flipped_outputs)
    if unbatched:
        flipped_outputs = self.backend.numpy.squeeze(flipped_outputs, axis=0)
    return flipped_outputs