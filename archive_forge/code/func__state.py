import threading
import sys
import tempfile
import time
from . import context
from . import process
from . import util
@_state.setter
def _state(self, value):
    self._array[0] = value