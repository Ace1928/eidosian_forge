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
class AbstractDecoder(object):
    protoComponent = None

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        raise error.PyAsn1Error('Decoder not implemented for %s' % (tagSet,))

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        raise error.PyAsn1Error('Indefinite length mode decoder not implemented for %s' % (tagSet,))