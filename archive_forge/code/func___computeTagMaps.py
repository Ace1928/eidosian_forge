import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def __computeTagMaps(self, unique):
    presentTypes = {}
    skipTypes = {}
    defaultType = None
    for namedType in self.__namedTypes:
        tagMap = namedType.asn1Object.tagMap
        if isinstance(tagMap, NamedTypes.PostponedError):
            return tagMap
        for tagSet in tagMap:
            if unique and tagSet in presentTypes:
                return NamedTypes.PostponedError('Non-unique tagSet %s of %s at %s' % (tagSet, namedType, self))
            presentTypes[tagSet] = namedType.asn1Object
        skipTypes.update(tagMap.skipTypes)
        if defaultType is None:
            defaultType = tagMap.defaultType
        elif tagMap.defaultType is not None:
            return NamedTypes.PostponedError('Duplicate default ASN.1 type at %s' % (self,))
    return tagmap.TagMap(presentTypes, skipTypes, defaultType)