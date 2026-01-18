from __future__ import annotations
import sys
import eventlet
def _green_subprocess_modules():
    from eventlet.green import subprocess
    return [('subprocess', subprocess)]