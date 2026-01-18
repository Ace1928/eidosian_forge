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
class RealPayloadDecoder(AbstractSimplePayloadDecoder):
    protoComponent = univ.Real()

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        for chunk in readFromStream(substrate, length, options):
            if isinstance(chunk, SubstrateUnderrunError):
                yield chunk
        if not chunk:
            yield self._createComponent(asn1Spec, tagSet, 0.0, **options)
            return
        fo = oct2int(chunk[0])
        chunk = chunk[1:]
        if fo & 128:
            if not chunk:
                raise error.PyAsn1Error('Incomplete floating-point value')
            if LOG:
                LOG('decoding binary encoded REAL')
            n = (fo & 3) + 1
            if n == 4:
                n = oct2int(chunk[0])
                chunk = chunk[1:]
            eo, chunk = (chunk[:n], chunk[n:])
            if not eo or not chunk:
                raise error.PyAsn1Error('Real exponent screwed')
            e = oct2int(eo[0]) & 128 and -1 or 0
            while eo:
                e <<= 8
                e |= oct2int(eo[0])
                eo = eo[1:]
            b = fo >> 4 & 3
            if b > 2:
                raise error.PyAsn1Error('Illegal Real base')
            if b == 1:
                e *= 3
            elif b == 2:
                e *= 4
            p = 0
            while chunk:
                p <<= 8
                p |= oct2int(chunk[0])
                chunk = chunk[1:]
            if fo & 64:
                p = -p
            sf = fo >> 2 & 3
            p *= 2 ** sf
            value = (p, 2, e)
        elif fo & 64:
            if LOG:
                LOG('decoding infinite REAL')
            value = fo & 1 and '-inf' or 'inf'
        elif fo & 192 == 0:
            if not chunk:
                raise error.PyAsn1Error('Incomplete floating-point value')
            if LOG:
                LOG('decoding character encoded REAL')
            try:
                if fo & 3 == 1:
                    value = (int(chunk), 10, 0)
                elif fo & 3 == 2:
                    value = float(chunk)
                elif fo & 3 == 3:
                    value = float(chunk)
                else:
                    raise error.SubstrateUnderrunError('Unknown NR (tag %s)' % fo)
            except ValueError:
                raise error.SubstrateUnderrunError('Bad character Real syntax')
        else:
            raise error.SubstrateUnderrunError('Unknown encoding (tag %s)' % fo)
        yield self._createComponent(asn1Spec, tagSet, value, **options)