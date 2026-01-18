from __future__ import annotations
import sys
import eventlet
def _green_time_modules():
    from eventlet.green import time
    return [('time', time)]