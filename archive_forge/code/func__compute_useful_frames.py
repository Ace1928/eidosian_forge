import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _compute_useful_frames(tb, num):
    """Return a list of frames, which form a 'useful' stack.

  Starting from the defining frame to the outermost one, this method computes
  the contiguous portion of the 'useful' stack trace and returns the selected
  frames.

  Args:
    tb: A list of traceback frames (as from Operation.traceback).
    num: total number of frames to return.

  Returns:
    A list of frames.
  """
    defining_frame_index = _find_index_of_defining_frame(tb)
    innermost_excluded = min(defining_frame_index + 2 + 1, len(tb))
    outermost_included = max(innermost_excluded - num, 0)
    return tb[outermost_included:innermost_excluded]