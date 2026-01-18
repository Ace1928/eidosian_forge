from collections import OrderedDict
from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class SetEncoder(AbstractItemEncoder):
    protoDict = dict

    def encode(self, value, encodeFun, **options):
        inconsistency = value.isInconsistent
        if inconsistency:
            raise inconsistency
        namedTypes = value.componentType
        substrate = self.protoDict()
        for idx, (key, subValue) in enumerate(value.items()):
            if namedTypes and namedTypes[idx].isOptional and (not value[idx].isValue):
                continue
            substrate[key] = encodeFun(subValue, **options)
        return substrate