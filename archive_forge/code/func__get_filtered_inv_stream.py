from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _get_filtered_inv_stream(self, revision_ids):
    from_repo = self.from_repository
    parent_ids = from_repo._find_parent_ids_of_revisions(revision_ids)
    parent_keys = [(p,) for p in parent_ids]
    find_text_keys = from_repo._serializer._find_text_key_references
    parent_text_keys = set(find_text_keys(from_repo._inventory_xml_lines_for_keys(parent_keys)))
    content_text_keys = set()
    knit = KnitVersionedFiles(None, None)
    factory = KnitPlainFactory()

    def find_text_keys_from_content(record):
        if record.storage_kind not in ('knit-delta-gz', 'knit-ft-gz'):
            raise ValueError('Unknown content storage kind for inventory text: %s' % (record.storage_kind,))
        raw_data = record._raw_record
        revision_id = record.key[-1]
        content, _ = knit._parse_record(revision_id, raw_data)
        if record.storage_kind == 'knit-delta-gz':
            line_iterator = factory.get_linedelta_content(content)
        elif record.storage_kind == 'knit-ft-gz':
            line_iterator = factory.get_fulltext_content(content)
        content_text_keys.update(find_text_keys([(line, revision_id) for line in line_iterator]))
    revision_keys = [(r,) for r in revision_ids]

    def _filtered_inv_stream():
        source_vf = from_repo.inventories
        stream = source_vf.get_record_stream(revision_keys, 'unordered', False)
        for record in stream:
            if record.storage_kind == 'absent':
                raise errors.NoSuchRevision(from_repo, record.key)
            find_text_keys_from_content(record)
            yield record
        self._text_keys = content_text_keys - parent_text_keys
    return ('inventories', _filtered_inv_stream())