import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class OctetStringPayloadDecoder(AbstractSimplePayloadDecoder):
    protoComponent = univ.OctetString('')
    supportConstructedForm = True

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun:
            asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        if tagSet[0].tagFormat == tag.tagFormatSimple:
            for chunk in readFromStream(substrate, length, options):
                if isinstance(chunk, SubstrateUnderrunError):
                    yield chunk
            yield self._createComponent(asn1Spec, tagSet, chunk, **options)
            return
        if not self.supportConstructedForm:
            raise error.PyAsn1Error('Constructed encoding form prohibited at %s' % self.__class__.__name__)
        if LOG:
            LOG('assembling constructed serialization')
        substrateFun = self.substrateCollector
        header = null
        original_position = substrate.tell()
        while substrate.tell() - original_position < length:
            for component in decodeFun(substrate, self.protoComponent, substrateFun=substrateFun, **options):
                if isinstance(component, SubstrateUnderrunError):
                    yield component
            header += component
        yield self._createComponent(asn1Spec, tagSet, header, **options)

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun and substrateFun is not self.substrateCollector:
            asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        substrateFun = self.substrateCollector
        header = null
        while True:
            for component in decodeFun(substrate, self.protoComponent, substrateFun=substrateFun, allowEoo=True, **options):
                if isinstance(component, SubstrateUnderrunError):
                    yield component
                if component is eoo.endOfOctets:
                    break
            if component is eoo.endOfOctets:
                break
            header += component
        yield self._createComponent(asn1Spec, tagSet, header, **options)