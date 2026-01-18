import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def _init_event_loop_policy():
    global _event_loop_policy
    with _lock:
        if _event_loop_policy is None:
            from . import DefaultEventLoopPolicy
            _event_loop_policy = DefaultEventLoopPolicy()