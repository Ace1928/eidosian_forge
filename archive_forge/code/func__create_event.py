import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _create_event(self, ph, category, name, pid, tid, timestamp):
    """Creates a new Chrome Trace event.

    For details of the file format, see:
    https://github.com/catapult-project/catapult/blob/master/tracing/README.md

    Args:
      ph:  The type of event - usually a single character.
      category: The event category as a string.
      name:  The event name as a string.
      pid:  Identifier of the process generating this event as an integer.
      tid:  Identifier of the thread generating this event as an integer.
      timestamp:  The timestamp of this event as a long integer.

    Returns:
      A JSON compatible event object.
    """
    event = {}
    event['ph'] = ph
    event['cat'] = category
    event['name'] = name
    event['pid'] = pid
    event['tid'] = tid
    event['ts'] = timestamp
    return event