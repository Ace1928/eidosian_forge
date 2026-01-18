import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _summaries(eval_dir):
    """Yields `tensorflow.Event` protos from event files in the eval dir.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Yields:
    `tensorflow.Event` object read from the event files.
  """
    if tf.compat.v1.gfile.Exists(eval_dir):
        for event_file in tf.compat.v1.gfile.Glob(os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            try:
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    yield event
            except tf.errors.DataLossError as e:
                tf.compat.v1.logging.warning('Skipping rest of the file due to encountering data corruption error; file path: %s; original error raised by `tf.train.summary_iterator`: %s', event_file, e)