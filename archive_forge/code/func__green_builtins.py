from __future__ import annotations
import sys
import eventlet
def _green_builtins():
    try:
        from eventlet.green import builtin
        return [('builtins', builtin)]
    except ImportError:
        return []