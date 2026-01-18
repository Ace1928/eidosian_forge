import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
def _saveTemp(self, filename, dumpFunc):
    with open(filename, 'wb') as f:
        dumpFunc(self.original, f)