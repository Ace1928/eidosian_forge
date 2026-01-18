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
class KnitIndexUnknownMethod(KnitError):
    """Raised when we don't understand the storage method.

    Currently only 'fulltext' and 'line-delta' are supported.
    """
    _fmt = 'Knit index %(filename)s does not have a known method in options: %(options)r'

    def __init__(self, filename, options):
        KnitError.__init__(self)
        self.filename = filename
        self.options = options