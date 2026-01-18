import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
def _on_call_done(self, future):
    self.resume()