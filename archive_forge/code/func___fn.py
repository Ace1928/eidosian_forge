from suds import *
from logging import getLogger
from suds.xsd.sxbasic import Factory as SXFactory
from suds.xsd.sxbasic import Attribute as SXAttribute
def __fn(x, y):
    ns = (None, 'http://schemas.xmlsoap.org/wsdl/')
    aty = y.get('arrayType', ns=ns)
    if aty is None:
        return SXAttribute(x, y)
    return Attribute(x, y, aty)