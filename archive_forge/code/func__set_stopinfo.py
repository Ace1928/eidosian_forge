import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def _set_stopinfo(self, stopframe, returnframe, stoplineno=0):
    """Set the attributes for stopping.

        If stoplineno is greater than or equal to 0, then stop at line
        greater than or equal to the stopline.  If stoplineno is -1, then
        don't stop at all.
        """
    self.stopframe = stopframe
    self.returnframe = returnframe
    self.quitting = False
    self.stoplineno = stoplineno