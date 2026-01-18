from ._icu_ import *
class ICUError(Exception):
    messages = {}

    def __str__(self):
        return '%s, error code: %d' % (self.args[1], self.args[0])

    def getErrorCode(self):
        return self.args[0]