import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
def _encodeComponents(self, value, asn1Spec, encodeFun, **options):
    if asn1Spec is None:
        inconsistency = value.isInconsistent
        if inconsistency:
            raise inconsistency
    else:
        asn1Spec = asn1Spec.componentType
    chunks = []
    wrapType = options.pop('wrapType', None)
    for idx, component in enumerate(value):
        chunk = encodeFun(component, asn1Spec, **options)
        if wrapType is not None and (not wrapType.isSameTypeWith(component)):
            chunk = encodeFun(chunk, wrapType, **options)
            if LOG:
                LOG('wrapped with wrap type %r' % (wrapType,))
        chunks.append(chunk)
    return chunks