import hashlib
import sys
from pyasn1.codec.der import decoder
from pyasn1.codec.der import encoder
from pyasn1.type import univ
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc2459
from pyasn1_modules import pem
class ValueOnlyBitStringEncoder(encoder.encoder.BitStringEncoder):

    def encodeTag(self, *args):
        return ''

    def encodeLength(self, *args):
        return ''

    def encodeValue(*args):
        substrate, isConstructed = encoder.encoder.BitStringEncoder.encodeValue(*args)
        return (substrate[1:], isConstructed)

    def __call__(self, bitStringValue):
        return self.encode(None, bitStringValue, defMode=True, maxChunkSize=0)