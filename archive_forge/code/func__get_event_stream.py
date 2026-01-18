from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _get_event_stream(self, shape):
    """Returns the event stream member's shape if any or None otherwise."""
    if shape is None:
        return None
    event_name = shape.event_stream_name
    if event_name:
        return shape.members[event_name]
    return None