import sys
class ExceptionInfo:

    def __init__(self, *info):
        if not info:
            info = sys.exc_info()
        self.type, self.value, _ = info

    def __bool__(self):
        """
        Return True if an exception occurred
        """
        return bool(self.type)
    __nonzero__ = __bool__