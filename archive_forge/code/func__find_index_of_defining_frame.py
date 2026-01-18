import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _find_index_of_defining_frame(tb):
    """Return index in op.traceback with first 'useful' frame.

  This method reads through the stack stored in op.traceback looking for the
  innermost frame which (hopefully) belongs to the caller.  It accomplishes this
  by rejecting frames deemed to be part of the TensorFlow framework (by
  pattern matching the filename).

  Args:
    tb: A list of traceback frames (as from Operation.traceback).

  Returns:
    Integer index into op.traceback where the first non-TF file was found
    (innermost to outermost), or 0 (for the outermost stack frame) if all files
    came from TensorFlow.
  """
    size = len(tb)
    filenames = [frame.filename for frame in tb]
    for idx, filename in enumerate(reversed(filenames)):
        is_framework = _is_framework_filename(filename)
        if not is_framework:
            return size - idx - 1
    return 0