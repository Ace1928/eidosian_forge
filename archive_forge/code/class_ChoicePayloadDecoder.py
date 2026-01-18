from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class ChoicePayloadDecoder(object):

    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        asn1Value = asn1Spec.clone()
        componentsTypes = asn1Spec.componentType
        for field in pyObject:
            if field in componentsTypes:
                asn1Value[field] = decodeFun(pyObject[field], componentsTypes[field].asn1Object, **options)
                break
        return asn1Value