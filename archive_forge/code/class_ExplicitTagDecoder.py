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
class ExplicitTagDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Any('')

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, '', **options), substrate, length)
        head, tail = (substrate[:length], substrate[length:])
        value, _ = decodeFun(head, asn1Spec, tagSet, length, **options)
        return (value, tail)

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, '', **options), substrate, length)
        value, substrate = decodeFun(substrate, asn1Spec, tagSet, length, **options)
        eooMarker, substrate = decodeFun(substrate, allowEoo=True, **options)
        if eooMarker is eoo.endOfOctets:
            return (value, substrate)
        else:
            raise error.PyAsn1Error('Missing end-of-octets terminator')