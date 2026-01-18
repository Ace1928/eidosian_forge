import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
Closes the queue, causing any pending or future `put()` calls to fail.