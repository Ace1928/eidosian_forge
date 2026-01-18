from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute

        Get whether the I{ns} is to B{not} be normalized.

        @param ns: A namespace.
        @type ns: (p, u)
        @return: True if to be skipped.
        @rtype: boolean

        