import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
def _is_active(self):
    return self.call is None or self.call.is_active()