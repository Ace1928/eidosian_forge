import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
class Deref:

    def __init__(self, num):
        self.refnum = num

    def getSource(self):
        return 'Deref(%d)' % self.refnum
    __repr__ = getSource