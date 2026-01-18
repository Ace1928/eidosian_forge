import os
from ctypes import *
class MAPIError(WindowsError):

    def __init__(self, code):
        WindowsError.__init__(self)
        self.code = code

    def __str__(self):
        return 'MAPI error %d' % (self.code,)