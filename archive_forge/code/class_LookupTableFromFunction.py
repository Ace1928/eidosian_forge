import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
@implementer(ILookupTable)
class LookupTableFromFunction:

    def __init__(self, in_table_function):
        self.lookup = in_table_function