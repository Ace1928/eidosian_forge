from pyasn1 import debug
from pyasn1 import error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class AbstractScalarDecoder(object):

    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        return asn1Spec.clone(pyObject)