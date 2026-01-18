from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _change_inv_parents(self, stream, get_parents, all_revision_keys):
    """Adapt a record stream to reconcile the parents."""
    for record in stream:
        wanted_parents = get_parents(record.key)
        if wanted_parents and wanted_parents[0] not in all_revision_keys:
            chunks = record.get_bytes_as('chunked')
            yield ChunkedContentFactory(record.key, wanted_parents, record.sha1, chunks)
        else:
            adapted_record = AdapterFactory(record.key, wanted_parents, record)
            yield adapted_record
        self._reweave_step('adding inventories')