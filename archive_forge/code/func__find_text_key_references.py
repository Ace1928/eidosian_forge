from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def _find_text_key_references(self, line_iterator):
    """Core routine for extracting references to texts from inventories.

        This performs the translation of xml lines to revision ids.

        :param line_iterator: An iterator of lines, origin_version_id
        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. Note that if that revision_id was
            not part of the line_iterator's output then False will be given -
            even though it may actually refer to that key.
        """
    if not self.support_altered_by_hack:
        raise AssertionError('_find_text_key_references only supported for branches which store inventory as unnested xml, not on %r' % self)
    result = {}
    unescape_revid_cache = {}
    unescape_fileid_cache = {}
    search = self._file_ids_altered_regex.search
    unescape = _unescape_xml
    setdefault = result.setdefault
    for line, line_key in line_iterator:
        match = search(line)
        if match is None:
            continue
        file_id, revision_id = match.group('file_id', 'revision_id')
        try:
            revision_id = unescape_revid_cache[revision_id]
        except KeyError:
            unescaped = unescape(revision_id)
            unescape_revid_cache[revision_id] = unescaped
            revision_id = unescaped
        try:
            file_id = unescape_fileid_cache[file_id]
        except KeyError:
            unescaped = unescape(file_id)
            unescape_fileid_cache[file_id] = unescaped
            file_id = unescaped
        key = (file_id, revision_id)
        setdefault(key, False)
        if revision_id == line_key[-1]:
            result[key] = True
    return result