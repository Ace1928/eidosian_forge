from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class AttrGroupQuery(Query):
    """
    Schema query class that searches for attributeGroup references in the
    specified schema.

    """

    def execute(self, schema):
        result = schema.agrps.get(self.ref)
        if self.filter(result):
            result = None
        return self.result(result)