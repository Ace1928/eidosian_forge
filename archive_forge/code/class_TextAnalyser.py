import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
class TextAnalyser(InputAnalyser):

    def correct_shape(self):
        if len(self.shape) == 1:
            return True
        return len(self.shape) == 2 and self.shape[1] == 1

    def finalize(self):
        if not self.correct_shape():
            raise ValueError('Expect the data to TextInput to have shape (batch_size, 1), but got input shape {shape}.'.format(shape=self.shape))
        if self.dtype != tf.string:
            raise TypeError('Expect the data to TextInput to be strings, but got {type}.'.format(type=self.dtype))