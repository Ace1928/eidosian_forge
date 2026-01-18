from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XAny(XBuiltin):
    """Represents an XSD <xsd:any/> node."""

    def __init__(self, schema, name):
        XBuiltin.__init__(self, schema, name)
        self.nillable = False

    def get_child(self, name):
        child = XAny(self.schema, name)
        return (child, [])

    def any(self):
        return True