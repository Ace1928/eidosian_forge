import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _alloc_flow_id(self):
    """Allocate a flow Id."""
    flow_id = self._next_flow_id
    self._next_flow_id += 1
    return flow_id