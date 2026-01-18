import threading
import warnings
from oslo_utils import importutils
from oslo_utils import timeutils
A class that provides consistent eventlet/threading Event API.

    This wraps the eventlet.event.Event class to have the same API as
    the standard threading.Event object.
    