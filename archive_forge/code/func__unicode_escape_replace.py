import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def _unicode_escape_replace(match, _map=_xml_escape_map):
    """Replace a string of non-ascii, non XML safe characters with their escape

    This will escape both Standard XML escapes, like <>"', etc.
    As well as escaping non ascii characters, because ElementTree did.
    This helps us remain compatible to older versions of bzr. We may change
    our policy in the future, though.
    """
    try:
        return _map[match.group()]
    except KeyError:
        return '&#%d;' % ord(match.group())