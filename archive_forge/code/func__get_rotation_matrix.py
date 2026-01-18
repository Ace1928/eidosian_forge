import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _get_rotation_matrix(self, inputs):
    shape = self.backend.core.shape(inputs)
    if len(shape) == 4:
        if self.data_format == 'channels_last':
            batch_size = shape[0]
            image_height = shape[1]
            image_width = shape[2]
        else:
            batch_size = shape[1]
            image_height = shape[2]
            image_width = shape[3]
    else:
        batch_size = 1
        if self.data_format == 'channels_last':
            image_height = shape[0]
            image_width = shape[1]
        else:
            image_height = shape[1]
            image_width = shape[2]
    lower = self._factor[0] * 2.0 * self.backend.convert_to_tensor(np.pi)
    upper = self._factor[1] * 2.0 * self.backend.convert_to_tensor(np.pi)
    seed_generator = self._get_seed_generator(self.backend._backend)
    angle = self.backend.random.uniform(shape=(batch_size,), minval=lower, maxval=upper, seed=seed_generator)
    cos_theta = self.backend.numpy.cos(angle)
    sin_theta = self.backend.numpy.sin(angle)
    image_height = self.backend.core.cast(image_height, cos_theta.dtype)
    image_width = self.backend.core.cast(image_width, cos_theta.dtype)
    x_offset = (image_width - 1 - (cos_theta * (image_width - 1) - sin_theta * (image_height - 1))) / 2.0
    y_offset = (image_height - 1 - (sin_theta * (image_width - 1) + cos_theta * (image_height - 1))) / 2.0
    outputs = self.backend.numpy.concatenate([self.backend.numpy.cos(angle)[:, None], -self.backend.numpy.sin(angle)[:, None], x_offset[:, None], self.backend.numpy.sin(angle)[:, None], self.backend.numpy.cos(angle)[:, None], y_offset[:, None], self.backend.numpy.zeros((batch_size, 2))], axis=1)
    if len(shape) == 3:
        outputs = self.backend.numpy.squeeze(outputs, axis=0)
    return outputs