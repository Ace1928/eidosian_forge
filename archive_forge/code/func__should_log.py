import functools
import logging
import os
from .state import PartialState
@staticmethod
def _should_log(main_process_only):
    """Check if log should be performed"""
    state = PartialState()
    return not main_process_only or (main_process_only and state.is_main_process)