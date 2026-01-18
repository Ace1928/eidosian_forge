from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _randomly_adjust_brightness(self, images):
    images_shape = self.backend.shape(images)
    rank = len(images_shape)
    if rank == 3:
        rgb_delta_shape = (1, 1, 1)
    elif rank == 4:
        rgb_delta_shape = [images_shape[0], 1, 1, 1]
    else:
        raise ValueError(f'Expected the input image to be rank 3 or 4. Received inputs.shape={images_shape}')
    seed_generator = self._get_seed_generator(self.backend._backend)
    rgb_delta = self.backend.random.uniform(minval=self._factor[0], maxval=self._factor[1], shape=rgb_delta_shape, seed=seed_generator)
    rgb_delta = rgb_delta * (self.value_range[1] - self.value_range[0])
    rgb_delta = self.backend.cast(rgb_delta, images.dtype)
    images += rgb_delta
    return self.backend.numpy.clip(images, self.value_range[0], self.value_range[1])