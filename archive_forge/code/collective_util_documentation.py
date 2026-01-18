import copy
import enum
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Creates a CollectiveHints.

    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This only applies to all-reduce with
        `MultiWorkerMirroredStrategy` currently.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. This can be useful when debugging hanging issues.
        This should only be used for debugging since it creates a new thread for
        each collective, i.e. an overhead of `timeout_seconds *
        num_collectives_per_second` more threads.  This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

    Raises:
      ValueError: When arguments have invalid value.
    