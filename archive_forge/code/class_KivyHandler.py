import gc
import weakref
import pytest
class KivyHandler(ExceptionHandler):

    def handle_exception(self, e):
        nonlocal exception
        exception = str(e)
        return kivy_exception_manager.PASS