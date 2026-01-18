import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
def _unload_code(self):
    """Called when done running code"""
    self.source = None
    self.code_context = None
    self.code_is_waiting = False