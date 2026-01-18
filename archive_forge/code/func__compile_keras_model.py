import keras_tuner
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks as blocks_module
from autokeras import keras_layers
from autokeras import nodes as nodes_module
from autokeras.engine import head as head_module
from autokeras.engine import serializable
from autokeras.utils import io_utils
def _compile_keras_model(self, hp, model):
    optimizer_name = hp.Choice('optimizer', ['adam', 'sgd', 'adam_weight_decay'], default='adam')
    learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001, 2e-05, 1e-05], default=0.001)
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'adam_weight_decay':
        steps_per_epoch = int(self.num_samples / self.batch_size)
        num_train_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.epochs * self.num_samples * 0.1 / self.batch_size)
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=num_train_steps, end_learning_rate=0.0)
        if warmup_steps:
            lr_schedule = keras_layers.WarmUp(initial_learning_rate=learning_rate, decay_schedule_fn=lr_schedule, warmup_steps=warmup_steps)
        optimizer = keras_layers.AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    model.compile(optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss())
    return model