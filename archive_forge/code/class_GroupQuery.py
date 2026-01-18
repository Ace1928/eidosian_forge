from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class GroupQuery(Query):
    """
    Schema query class that searches for Group references in the specified
    schema.

    """

    def execute(self, schema):
        result = schema.groups.get(self.ref)
        if self.filter(result):
            result = None
        return self.result(result)