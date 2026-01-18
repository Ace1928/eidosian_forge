import collections
import logging
import threading
import time
import types
def __get_result(self):
    if self._exception:
        try:
            raise self._exception
        finally:
            self = None
    else:
        return self._result