from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def _unescaper(match, _map=_xml_unescape_map):
    code = match.group(1)
    try:
        return _map[code]
    except KeyError:
        if not code.startswith(b'#'):
            raise
        return chr(int(code[1:])).encode('utf8')