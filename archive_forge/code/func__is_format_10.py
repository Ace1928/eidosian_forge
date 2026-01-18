from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
def _is_format_10(value):
    if value != 10:
        raise ValueError('Format number was not recognized, expected 10 got %d' % (value,))
    return 10