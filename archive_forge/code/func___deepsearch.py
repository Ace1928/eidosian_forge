from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
def __deepsearch(self, schema):
    from suds.xsd.sxbasic import Element
    result = None
    for e in schema.all:
        result = e.find(self.ref, (Element,))
        if self.filter(result):
            result = None
        else:
            break
    return result