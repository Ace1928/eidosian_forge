import pytest
import rpy2.rinterface
import rpy2.rinterface_lib.embedded
from threading import Thread
from rpy2.rinterface import embedded
class ThreadWithExceptions(Thread):
    """Wrapper around Thread allowing to record exceptions from the thread."""

    def run(self):
        self.exception = None
        try:
            self._target()
        except Exception as e:
            self.exception = e