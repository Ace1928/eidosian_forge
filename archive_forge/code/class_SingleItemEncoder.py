from collections import OrderedDict
from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class SingleItemEncoder(object):
    TAG_MAP = TAG_MAP
    TYPE_MAP = TYPE_MAP

    def __init__(self, tagMap=_MISSING, typeMap=_MISSING, **ignored):
        self._tagMap = tagMap if tagMap is not _MISSING else self.TAG_MAP
        self._typeMap = typeMap if typeMap is not _MISSING else self.TYPE_MAP

    def __call__(self, value, **options):
        if not isinstance(value, base.Asn1Item):
            raise error.PyAsn1Error('value is not valid (should be an instance of an ASN.1 Item)')
        if LOG:
            debug.scope.push(type(value).__name__)
            LOG('encoder called for type %s <%s>' % (type(value).__name__, value.prettyPrint()))
        tagSet = value.tagSet
        try:
            concreteEncoder = self._typeMap[value.typeId]
        except KeyError:
            baseTagSet = tag.TagSet(value.tagSet.baseTag, value.tagSet.baseTag)
            try:
                concreteEncoder = self._tagMap[baseTagSet]
            except KeyError:
                raise error.PyAsn1Error('No encoder for %s' % (value,))
        if LOG:
            LOG('using value codec %s chosen by %s' % (concreteEncoder.__class__.__name__, tagSet))
        pyObject = concreteEncoder.encode(value, self, **options)
        if LOG:
            LOG('encoder %s produced: %s' % (type(concreteEncoder).__name__, repr(pyObject)))
            debug.scope.pop()
        return pyObject