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
class AbstractSimplePayloadDecoder(AbstractPayloadDecoder):

    @staticmethod
    def substrateCollector(asn1Object, substrate, length, options):
        for chunk in readFromStream(substrate, length, options):
            yield chunk

    def _createComponent(self, asn1Spec, tagSet, value, **options):
        if options.get('native'):
            return value
        elif asn1Spec is None:
            return self.protoComponent.clone(value, tagSet=tagSet)
        elif value is noValue:
            return asn1Spec
        else:
            return asn1Spec.clone(value)