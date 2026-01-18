from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class BlindQuery(Query):
    """
    Schema query class that I{blindly} searches for a reference in the
    specified schema. It may be used to find Elements and Types but will match
    on an Element first. This query will also find builtins.

    """

    def execute(self, schema):
        if schema.builtin(self.ref):
            name = self.ref[0]
            b = Factory.create(schema, name)
            log.debug('%s, found builtin (%s)', self.id, name)
            return b
        result = None
        for d in (schema.elements, schema.types):
            result = d.get(self.ref)
            if self.filter(result):
                result = None
            else:
                break
        if result is None:
            eq = ElementQuery(self.ref)
            eq.history = self.history
            result = eq.execute(schema)
        return self.result(result)