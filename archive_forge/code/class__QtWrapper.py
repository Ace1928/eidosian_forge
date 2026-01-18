import logging
import sys
from .indenter import write_code
from .qtproxies import QtGui, QtWidgets, Literal, strict_getattr
class _QtWrapper(object):

    @classmethod
    def search(cls, name):
        try:
            return strict_getattr(cls.module, name)
        except AttributeError:
            return None