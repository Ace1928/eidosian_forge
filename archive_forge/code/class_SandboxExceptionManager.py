from functools import wraps
from kivy.context import Context
from kivy.base import ExceptionManagerBase
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
class SandboxExceptionManager(ExceptionManagerBase):

    def __init__(self, sandbox):
        ExceptionManagerBase.__init__(self)
        self.sandbox = sandbox

    def handle_exception(self, e):
        if not self.sandbox.on_exception(e):
            return ExceptionManagerBase.RAISE
        return ExceptionManagerBase.PASS