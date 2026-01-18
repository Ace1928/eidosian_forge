import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _get_error_message(self):
    """Get the output messages produced since the last reset as
        one string. Returns 'No known reason.' if there are no messages.
        Also resets the log.
        """
    if self._messages:
        res = ' '.join(self._messages)
        self._reset_log()
        return res
    else:
        return 'No known reason.'