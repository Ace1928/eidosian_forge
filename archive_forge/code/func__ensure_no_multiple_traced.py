import collections
import datetime
import functools
import inspect
import socket
import threading
from oslo_utils import reflection
from oslo_utils import uuidutils
from osprofiler import _utils as utils
from osprofiler import notifier
def _ensure_no_multiple_traced(traceable_attrs):
    for attr_name, attr in traceable_attrs:
        traced_times = getattr(attr, '__traced__', 0)
        if traced_times:
            raise ValueError("Can not apply new trace on top of previously traced attribute '%s' since it has been traced %s times previously" % (attr_name, traced_times))