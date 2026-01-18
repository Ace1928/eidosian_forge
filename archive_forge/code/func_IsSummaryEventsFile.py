import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def IsSummaryEventsFile(path):
    """Check whether the path is probably a TF Events file containing Summary.

    Args:
      path: A file path to check if it is an event file containing `Summary`
        protos.

    Returns:
      If path is formatted like a TensorFlowEventsFile. Dummy files such as
        those created with the '.profile-empty' suffixes and meant to hold
        no `Summary` protos  are treated as `False`. For background, see:
        https://github.com/tensorflow/tensorboard/issues/2084.
    """
    return IsTensorFlowEventsFile(path) and (not path.endswith('.profile-empty'))