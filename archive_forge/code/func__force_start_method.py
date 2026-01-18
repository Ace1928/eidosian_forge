import os
import sys
import threading
from . import process
from . import reduction
def _force_start_method(method):
    _default_context._actual_context = _concrete_contexts[method]