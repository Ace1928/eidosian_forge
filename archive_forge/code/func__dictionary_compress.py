import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _dictionary_compress(self, keys):
    """Dictionary compress keys.

        :param keys: The keys to generate references to.
        :return: A string representation of keys. keys which are present are
            dictionary compressed, and others are emitted as fulltext with a
            '.' prefix.
        """
    if not keys:
        return b''
    result_list = []
    prefix = keys[0][:-1]
    cache = self._kndx_cache[prefix][0]
    for key in keys:
        if key[:-1] != prefix:
            raise ValueError('mismatched prefixes for %r' % keys)
        if key[-1] in cache:
            result_list.append(b'%d' % cache[key[-1]][5])
        else:
            result_list.append(b'.' + key[-1])
    return b' '.join(result_list)