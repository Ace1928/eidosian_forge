from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def _version2fieldlist(version):
    if version == '1.0':
        return _241_FIELDS
    elif version == '1.1':
        return _314_FIELDS
    elif version == '1.2':
        return _345_FIELDS
    elif version in ('1.3', '2.1'):
        return _345_FIELDS + tuple((f for f in _566_FIELDS if f not in _345_FIELDS))
    elif version == '2.0':
        raise ValueError('Metadata 2.0 is withdrawn and not supported')
    elif version == '2.2':
        return _643_FIELDS
    raise MetadataUnrecognizedVersionError(version)