from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from mlflow import log_metrics, log_params, log_text
from mlflow.utils.autologging_utils import ExceptionSafeClass
from mlflow.utils.checkpoint_utils import MlflowModelCheckpointCallbackBase
class MlflowModelCheckpointCallback(Callback, MlflowModelCheckpointCallbackBase):
    """Callback for automatic Keras model checkpointing to MLflow.

    Args:
        monitor: In automatic model checkpointing, the metric name to monitor if
            you set `model_checkpoint_save_best_only` to True.
        save_best_only: If True, automatic model checkpointing only saves when
            the model is considered the "best" model according to the quantity
            monitored and previous checkpoint model is overwritten.
        mode: one of {"min", "max"}. In automatic model checkpointing,
            if save_best_only=True, the decision to overwrite the current save file is made
            based on either the maximization or the minimization of the monitored quantity.
        save_weights_only: In automatic model checkpointing, if True, then
            only the model’s weights will be saved. Otherwise, the optimizer states,
            lr-scheduler states, etc are added in the checkpoint too.
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. Note that if the saving isn't
            aligned to epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.

    .. code-block:: python
        :caption: Example

        from tensorflow import keras
        import tensorflow as tf
        import mlflow
        import numpy as np
        from mlflow.tensorflow import MlflowModelCheckpointCallback

        # Prepare data for a 2-class classification.
        data = tf.random.uniform([8, 28, 28, 3])
        label = tf.convert_to_tensor(np.random.randint(2, size=8))

        model = keras.Sequential(
            [
                keras.Input([28, 28, 3]),
                keras.layers.Flatten(),
                keras.layers.Dense(2),
            ]
        )

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(0.001),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        mlflow_checkpoint_callback = MlflowModelCheckpointCallback(
            monitor="sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )

        with mlflow.start_run() as run:
            model.fit(
                data,
                label,
                batch_size=4,
                epochs=2,
                callbacks=[mlflow_checkpoint_callback],
            )
    """

    def __init__(self, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, save_freq='epoch'):
        Callback.__init__(self)
        MlflowModelCheckpointCallbackBase.__init__(self, checkpoint_file_suffix='.h5', monitor=monitor, mode=mode, save_best_only=save_best_only, save_weights_only=save_weights_only, save_freq=save_freq)
        self.trainer = None
        self.current_epoch = None
        self._last_batch_seen = 0
        self.global_step = 0
        self.global_step_last_saving = 0

    def save_checkpoint(self, filepath: str):
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        add_batches = batch + 1 if batch <= self._last_batch_seen else batch - self._last_batch_seen
        self._last_batch_seen = batch
        self.global_step += add_batches
        if isinstance(self.save_freq, int):
            if self.global_step - self.global_step_last_saving >= self.save_freq:
                self.check_and_save_checkpoint_if_needed(current_epoch=self.current_epoch, global_step=self.global_step, metric_dict={k: float(v) for k, v in logs.items()})
                self.global_step_last_saving = self.global_step

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == 'epoch':
            self.check_and_save_checkpoint_if_needed(current_epoch=self.current_epoch, global_step=self.global_step, metric_dict={k: float(v) for k, v in logs.items()})