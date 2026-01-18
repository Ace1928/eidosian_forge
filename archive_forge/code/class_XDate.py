from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XDate(XBuiltin):
    """Represents an XSD <xsd:date/> built-in type."""

    @staticmethod
    def translate(value, topython=True):
        if topython:
            if isinstance(value, str) and value:
                return Date(value).value
        else:
            if isinstance(value, datetime.date):
                return Date(value)
            return value