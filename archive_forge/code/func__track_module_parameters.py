import io
from packaging.version import parse
from keras.src.api_export import keras_export
from keras.src.layers import Layer
from keras.src.ops import convert_to_numpy
from keras.src.ops import convert_to_tensor
def _track_module_parameters(self):
    from keras.src.backend.torch import Variable
    for param in self.module.parameters():
        variable = Variable(initializer=param, trainable=param.requires_grad)
        self._track_variable(variable)
    self.built = True