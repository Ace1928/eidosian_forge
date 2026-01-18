import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
class AllocationMaximum(collections.namedtuple('AllocationMaximum', ('timestamp', 'num_bytes', 'tensors'))):
    """Stores the maximum allocation for a given allocator within the timelne.

  Parameters:
    timestamp: `tensorflow::Env::NowMicros()` when this maximum was reached.
    num_bytes: the total memory used at this time.
    tensors: the set of tensors allocated at this time.
  """
    pass