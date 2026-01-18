import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def __computeMinTagSet(self):
    minTagSet = None
    for namedType in self.__namedTypes:
        asn1Object = namedType.asn1Object
        try:
            tagSet = asn1Object.minTagSet
        except AttributeError:
            tagSet = asn1Object.tagSet
        if minTagSet is None or tagSet < minTagSet:
            minTagSet = tagSet
    return minTagSet or tag.TagSet()