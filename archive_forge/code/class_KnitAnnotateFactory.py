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
class KnitAnnotateFactory(_KnitFactory):
    """Factory for creating annotated Content objects."""
    annotated = True

    def make(self, lines, version_id):
        num_lines = len(lines)
        return AnnotatedKnitContent(zip([version_id] * num_lines, lines))

    def parse_fulltext(self, content, version_id):
        """Convert fulltext to internal representation

        fulltext content is of the format
        revid(utf8) plaintext

        internal representation is of the format:
        (revid, plaintext)
        """
        lines = (tuple(line.split(b' ', 1)) for line in content)
        return AnnotatedKnitContent(lines)

    def parse_line_delta(self, lines, version_id, plain=False):
        """Convert a line based delta into internal representation.

        line delta is in the form of:
        intstart intend intcount
        1..count lines:
        revid(utf8) newline

        internal representation is
        (start, end, count, [1..count tuples (revid, newline)])

        :param plain: If True, the lines are returned as a plain
            list without annotations, not as a list of (origin, content) tuples, i.e.
            (start, end, count, [1..count newline])
        """
        result = []
        lines = iter(lines)
        cache = {}

        def cache_and_return(line):
            origin, text = line.split(b' ', 1)
            return (cache.setdefault(origin, origin), text)
        if plain:
            for header in lines:
                start, end, count = (int(n) for n in header.split(b','))
                contents = [next(lines).split(b' ', 1)[1] for _ in range(count)]
                result.append((start, end, count, contents))
        else:
            for header in lines:
                start, end, count = (int(n) for n in header.split(b','))
                contents = [tuple(next(lines).split(b' ', 1)) for _ in range(count)]
                result.append((start, end, count, contents))
        return result

    def get_fulltext_content(self, lines):
        """Extract just the content lines from a fulltext."""
        return (line.split(b' ', 1)[1] for line in lines)

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
                origin, text = next(lines).split(b' ', 1)
                yield text

    def lower_fulltext(self, content):
        """convert a fulltext content record into a serializable form.

        see parse_fulltext which this inverts.
        """
        return [b'%s %s' % (o, t) for o, t in content._lines]

    def lower_line_delta(self, delta):
        """convert a delta into a serializable form.

        See parse_line_delta which this inverts.
        """
        out = []
        for start, end, c, lines in delta:
            out.append(b'%d,%d,%d\n' % (start, end, c))
            out.extend((origin + b' ' + text for origin, text in lines))
        return out

    def annotate(self, knit, key):
        content = knit._get_content(key)
        if isinstance(key, tuple):
            prefix = key[:-1]
            origins = content.annotate()
            result = []
            for origin, line in origins:
                result.append((prefix + (origin,), line))
            return result
        else:
            return content.annotate()