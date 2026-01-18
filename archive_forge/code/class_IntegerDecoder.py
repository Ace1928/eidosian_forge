from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class IntegerDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Integer(0)

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        head, tail = (substrate[:length], substrate[length:])
        if not head:
            return (self._createComponent(asn1Spec, tagSet, 0, **options), tail)
        value = from_bytes(head, signed=True)
        return (self._createComponent(asn1Spec, tagSet, value, **options), tail)