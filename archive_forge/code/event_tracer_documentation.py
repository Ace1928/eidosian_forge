import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
 The traits post event tracer.

        This method should be set as the global post event tracer for traits.

        