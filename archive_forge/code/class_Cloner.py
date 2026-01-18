import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class Cloner(object):
    """An object used to clone other objects."""

    def clone(cls, original):
        """Return an exact copy of an object."""
        'The original object must have an empty constructor.'
        return cls.create(original.__class__)

    def create(cls, type):
        """Create an object of a given class."""
        clone = type.__new__(type)
        clone.__init__()
        return clone
    clone = classmethod(clone)
    create = classmethod(create)