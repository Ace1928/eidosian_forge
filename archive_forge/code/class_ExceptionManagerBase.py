import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
class ExceptionManagerBase:
    """ExceptionManager manages exceptions handlers."""
    RAISE = 0
    'The exception should be re-raised.\n    '
    PASS = 1
    'The exception should be ignored as it was handled by the handler.\n    '

    def __init__(self):
        self.handlers = []
        self.policy = ExceptionManagerBase.RAISE

    def add_handler(self, cls):
        """Add a new exception handler to the stack."""
        if cls not in self.handlers:
            self.handlers.append(cls)

    def remove_handler(self, cls):
        """Remove the exception handler from the stack."""
        if cls in self.handlers:
            self.handlers.remove(cls)

    def handle_exception(self, inst):
        """Called when an exception occurred in the :func:`runTouchApp`
        main loop."""
        ret = self.policy
        for handler in self.handlers:
            r = handler.handle_exception(inst)
            if r == ExceptionManagerBase.PASS:
                ret = r
        return ret