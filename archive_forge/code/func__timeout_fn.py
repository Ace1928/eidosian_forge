import re
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import optimizer
from tensorflow.python.util.tf_export import keras_export
def _timeout_fn(self):
    logging.info(f'No checkpoints appear to be found after {_CHECKPOINT_TIMEOUT_SEC} seconds. Please check if you are properly using a `tf.train.Checkpoint/CheckpointManager` or `tf.keras.callbacks.ModelCheckpoint(save_weights_only=True)` to save checkpoints by the training. See `tf.keras.SidecarEvaluator` doc for recommended flows of saving checkpoints.')
    return False