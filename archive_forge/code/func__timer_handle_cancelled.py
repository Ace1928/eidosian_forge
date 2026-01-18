import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _timer_handle_cancelled(self, handle):
    """Notification that a TimerHandle has been cancelled."""
    raise NotImplementedError