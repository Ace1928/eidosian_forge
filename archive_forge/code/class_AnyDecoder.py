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
class AnyDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Any()

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if asn1Spec is None or (asn1Spec is not None and tagSet != asn1Spec.tagSet):
            fullSubstrate = options['fullSubstrate']
            length += len(fullSubstrate) - len(substrate)
            substrate = fullSubstrate
        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, noValue, **options), substrate, length)
        head, tail = (substrate[:length], substrate[length:])
        return (self._createComponent(asn1Spec, tagSet, head, **options), tail)

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if asn1Spec is not None and tagSet == asn1Spec.tagSet:
            header = null
        else:
            fullSubstrate = options['fullSubstrate']
            header = fullSubstrate[:-len(substrate)]
        asn1Spec = self.protoComponent
        if substrateFun and substrateFun is not self.substrateCollector:
            asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
            return substrateFun(asn1Object, header + substrate, length + len(header))
        substrateFun = self.substrateCollector
        while substrate:
            component, substrate = decodeFun(substrate, asn1Spec, substrateFun=substrateFun, allowEoo=True, **options)
            if component is eoo.endOfOctets:
                break
            header += component
        else:
            raise error.SubstrateUnderrunError('No EOO seen before substrate ends')
        if substrateFun:
            return (header, substrate)
        else:
            return (self._createComponent(asn1Spec, tagSet, header, **options), substrate)