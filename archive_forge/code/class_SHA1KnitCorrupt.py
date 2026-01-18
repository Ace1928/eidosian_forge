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
class SHA1KnitCorrupt(KnitCorrupt):
    _fmt = 'Knit %(filename)s corrupt: sha-1 of reconstructed text does not match expected sha-1. key %(key)s expected sha %(expected)s actual sha %(actual)s'

    def __init__(self, filename, actual, expected, key, content):
        KnitError.__init__(self)
        self.filename = filename
        self.actual = actual
        self.expected = expected
        self.key = key
        self.content = content