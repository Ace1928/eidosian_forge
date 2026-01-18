import sys
import warnings
from collections import deque
from functools import wraps
def _make_context_fixer(frame_exc):
    return lambda new_exc, old_exc: None