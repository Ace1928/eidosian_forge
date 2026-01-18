import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _reset_log(self):
    """Reset the list of output messages. Call this before
        loading or saving an image with the FreeImage API.
        """
    self._messages = []