from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def _unpack_revision_properties(self, elt, rev):
    """Unpack properties onto a revision."""
    props_elt = elt.find('properties')
    if props_elt is None:
        return
    for prop_elt in props_elt:
        if prop_elt.tag != 'property':
            raise AssertionError('bad tag under properties list: %r' % prop_elt.tag)
        name = prop_elt.get('name')
        value = prop_elt.text
        if value is None:
            value = ''
        if name in rev.properties:
            raise AssertionError('repeated property %r' % name)
        rev.properties[name] = value