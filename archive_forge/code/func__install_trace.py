import os
import sys
from typing import TYPE_CHECKING, Dict
def _install_trace():
    global _trace_thread_count
    sys.setprofile(_thread_trace_func(_trace_thread_count))
    _trace_thread_count += 1