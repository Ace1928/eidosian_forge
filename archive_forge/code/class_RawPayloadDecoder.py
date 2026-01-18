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
class RawPayloadDecoder(AbstractSimplePayloadDecoder):
    protoComponent = univ.Any('')

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun:
            asn1Object = self._createComponent(asn1Spec, tagSet, '', **options)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        for value in decodeFun(substrate, asn1Spec, tagSet, length, **options):
            yield value

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if substrateFun:
            asn1Object = self._createComponent(asn1Spec, tagSet, '', **options)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        while True:
            for value in decodeFun(substrate, asn1Spec, tagSet, length, allowEoo=True, **options):
                if value is eoo.endOfOctets:
                    return
                yield value