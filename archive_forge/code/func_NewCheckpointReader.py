from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.NewCheckpointReader'])
def NewCheckpointReader(filepattern):
    """A function that returns a CheckPointReader.

  Args:
    filepattern: The filename.

  Returns:
    A CheckpointReader object.
  """
    try:
        return CheckpointReader(compat.as_bytes(filepattern))
    except RuntimeError as e:
        error_translator(e)