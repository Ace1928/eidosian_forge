import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
class ILookupTable(Interface):
    """
    Interface for character lookup classes.
    """

    def lookup(c):
        """
        Return whether character is in this table.
        """