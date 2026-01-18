import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class SimpleVal(object):
    """
    A simple validator.
    Can be used to check that all members expected are present.
    
    To use it, provide a configspec with all your members in (the value given
    will be ignored). Pass an instance of ``SimpleVal`` to the ``validate``
    method of your ``ConfigObj``. ``validate`` will return ``True`` if all
    members are present, or a dictionary with True/False meaning
    present/missing. (Whole missing sections will be replaced with ``False``)
    """

    def __init__(self):
        self.baseErrorClass = ConfigObjError

    def check(self, check, member, missing=False):
        """A dummy check method, always returns the value unchanged."""
        if missing:
            raise self.baseErrorClass()
        return member