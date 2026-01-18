import tensorflow as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
from tensorboard.plugins.hparams import summary_v2
Create a callback for logging hyperparameters to TensorBoard.

        As with the standard `tf.keras.callbacks.TensorBoard` class, each
        callback object is valid for only one call to `model.fit`.

        Args:
          writer: The `SummaryWriter` object to which hparams should be
            written, or a logdir (as a `str`) to be passed to
            `tf.summary.create_file_writer` to create such a writer.
          hparams: A `dict` mapping hyperparameters to the values used in
            this session. Keys should be the names of `HParam` objects used
            in an experiment, or the `HParam` objects themselves. Values
            should be Python `bool`, `int`, `float`, or `string` values,
            depending on the type of the hyperparameter.
          trial_id: An optional `str` ID for the set of hyperparameter
            values used in this trial. Defaults to a hash of the
            hyperparameters.

        Raises:
          ValueError: If two entries in `hparams` share the same
            hyperparameter name.
        