import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
class ImageAnalyser(InputAnalyser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def finalize(self):
        if len(self.shape) not in [3, 4]:
            raise ValueError('Expect the data to ImageInput to have shape (batch_size, height, width, channels) or (batch_size, height, width) dimensions, but got input shape {shape}'.format(shape=self.shape))