from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def addPrefix(self, p, u):
    """
        Add or update a prefix mapping.

        @param p: A prefix.
        @type p: basestring
        @param u: A namespace URI.
        @type u: basestring
        @return: self
        @rtype: L{Element}
        """
    self.nsprefixes[p] = u
    return self