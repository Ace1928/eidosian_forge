from pyasn1 import debug
from pyasn1 import error
from pyasn1.compat import _MISSING
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class SingleItemDecoder(object):
    TAG_MAP = TAG_MAP
    TYPE_MAP = TYPE_MAP

    def __init__(self, tagMap=_MISSING, typeMap=_MISSING, **ignored):
        self._tagMap = tagMap if tagMap is not _MISSING else self.TAG_MAP
        self._typeMap = typeMap if typeMap is not _MISSING else self.TYPE_MAP

    def __call__(self, pyObject, asn1Spec, **options):
        if LOG:
            debug.scope.push(type(pyObject).__name__)
            LOG('decoder called at scope %s, working with type %s' % (debug.scope, type(pyObject).__name__))
        if asn1Spec is None or not isinstance(asn1Spec, base.Asn1Item):
            raise error.PyAsn1Error('asn1Spec is not valid (should be an instance of an ASN.1 Item, not %s)' % asn1Spec.__class__.__name__)
        try:
            valueDecoder = self._typeMap[asn1Spec.typeId]
        except KeyError:
            baseTagSet = tag.TagSet(asn1Spec.tagSet.baseTag, asn1Spec.tagSet.baseTag)
            try:
                valueDecoder = self._tagMap[baseTagSet]
            except KeyError:
                raise error.PyAsn1Error('Unknown ASN.1 tag %s' % asn1Spec.tagSet)
        if LOG:
            LOG('calling decoder %s on Python type %s <%s>' % (type(valueDecoder).__name__, type(pyObject).__name__, repr(pyObject)))
        value = valueDecoder(pyObject, asn1Spec, self, **options)
        if LOG:
            LOG('decoder %s produced ASN.1 type %s <%s>' % (type(valueDecoder).__name__, type(value).__name__, repr(value)))
            debug.scope.pop()
        return value