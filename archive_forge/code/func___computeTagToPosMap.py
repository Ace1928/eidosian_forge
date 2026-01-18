import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def __computeTagToPosMap(self):
    tagToPosMap = {}
    for idx, namedType in enumerate(self.__namedTypes):
        tagMap = namedType.asn1Object.tagMap
        if isinstance(tagMap, NamedTypes.PostponedError):
            return tagMap
        if not tagMap:
            continue
        for _tagSet in tagMap.presentTypes:
            if _tagSet in tagToPosMap:
                return NamedTypes.PostponedError('Duplicate component tag %s at %s' % (_tagSet, namedType))
            tagToPosMap[_tagSet] = idx
    return tagToPosMap