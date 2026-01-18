from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class AttrQuery(Query):
    """
    Schema query class that searches for Attribute references in the specified
    schema. Matches on root Attribute by qname first, then searches deeper into
    the document.

    """

    def execute(self, schema):
        result = schema.attributes.get(self.ref)
        if self.filter(result):
            result = self.__deepsearch(schema)
        return self.result(result)

    def __deepsearch(self, schema):
        from suds.xsd.sxbasic import Attribute
        result = None
        for e in schema.all:
            result = e.find(self.ref, (Attribute,))
            if self.filter(result):
                result = None
            else:
                break
        return result