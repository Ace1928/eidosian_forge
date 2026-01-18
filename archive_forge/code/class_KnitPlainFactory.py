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
class KnitPlainFactory(_KnitFactory):
    """Factory for creating plain Content objects."""
    annotated = False

    def make(self, lines, version_id):
        return PlainKnitContent(lines, version_id)

    def parse_fulltext(self, content, version_id):
        """This parses an unannotated fulltext.

        Note that this is not a noop - the internal representation
        has (versionid, line) - its just a constant versionid.
        """
        return self.make(content, version_id)

    def parse_line_delta_iter(self, lines, version_id):
        cur = 0
        num_lines = len(lines)
        while cur < num_lines:
            header = lines[cur]
            cur += 1
            start, end, c = (int(n) for n in header.split(b','))
            yield (start, end, c, lines[cur:cur + c])
            cur += c

    def parse_line_delta(self, lines, version_id):
        return list(self.parse_line_delta_iter(lines, version_id))

    def get_fulltext_content(self, lines):
        """Extract just the content lines from a fulltext."""
        return iter(lines)

    def get_linedelta_content(self, lines):
        """Extract just the content from a line delta.

        This doesn't return all of the extra information stored in a delta.
        Only the actual content lines.
        """
        lines = iter(lines)
        for header in lines:
            header = header.split(b',')
            count = int(header[2])
            for _ in range(count):
                yield next(lines)

    def lower_fulltext(self, content):
        return content.text()

    def lower_line_delta(self, delta):
        out = []
        for start, end, c, lines in delta:
            out.append(b'%d,%d,%d\n' % (start, end, c))
            out.extend(lines)
        return out

    def annotate(self, knit, key):
        annotator = _KnitAnnotator(knit)
        return annotator.annotate_flat(key)