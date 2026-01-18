from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
def _swap_variables(self):
    if hasattr(self.model.optimizer, 'inner_optimizer'):
        optimizer = self.model.optimizer.inner_optimizer
    else:
        optimizer = self.model.optimizer
    if not hasattr(optimizer, '_model_variables_moving_average'):
        raise ValueError(f'SwapEMAWeights must be used when `use_ema=True` is set on the optimizer. Received: use_ema={optimizer.use_ema}')
    if backend.backend() == 'tensorflow':
        self._tf_swap_variables(optimizer)
    else:
        self._backend_swap_variables(optimizer)