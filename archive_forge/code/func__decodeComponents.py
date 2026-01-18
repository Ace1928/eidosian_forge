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
def _decodeComponents(self, substrate, tagSet=None, decodeFun=None, **options):
    components = []
    componentTypes = set()
    while substrate:
        component, substrate = decodeFun(substrate, **options)
        if component is eoo.endOfOctets:
            break
        components.append(component)
        componentTypes.add(component.tagSet)
    if len(componentTypes) > 1:
        protoComponent = self.protoRecordComponent
    else:
        protoComponent = self.protoSequenceComponent
    asn1Object = protoComponent.clone(tagSet=tag.TagSet(protoComponent.tagSet.baseTag, *tagSet.superTags))
    for idx, component in enumerate(components):
        asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
    return (asn1Object, substrate)