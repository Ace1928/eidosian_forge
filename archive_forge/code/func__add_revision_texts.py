import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def _add_revision_texts(self, revision_order):
    parent_map = self.repository.get_parent_map(revision_order)
    revision_to_bytes = self.repository._serializer.write_revision_to_string
    revisions = self.repository.get_revisions(revision_order)
    for revision in revisions:
        revision_id = revision.revision_id
        parents = parent_map.get(revision_id, None)
        revision_text = revision_to_bytes(revision)
        self.bundle.add_fulltext_record(revision_text, parents, 'revision', revision_id)
        try:
            self.bundle.add_fulltext_record(self.repository.get_signature_text(revision_id), parents, 'signature', revision_id)
        except errors.NoSuchRevision:
            pass