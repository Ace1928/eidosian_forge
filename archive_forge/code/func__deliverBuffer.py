import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def _deliverBuffer(self, buf):
    if buf:
        for ch in iterbytes(buf[:-1]):
            self.characterReceived(ch, True)
        self.characterReceived(buf[-1:], False)