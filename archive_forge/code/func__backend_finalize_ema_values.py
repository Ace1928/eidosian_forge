from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
def _backend_finalize_ema_values(self, optimizer):
    for var, average_var in zip(self.model.trainable_variables, optimizer._model_variables_moving_average):
        average_var.assign(var)