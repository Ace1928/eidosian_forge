import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
def _inventory_delta_stream(self, repository, ordering, revids):
    prev_inv = _mod_inventory.Inventory(root_id=None, revision_id=_mod_revision.NULL_REVISION)
    serializer = inventory_delta.InventoryDeltaSerializer(repository.supports_rich_root(), repository._format.supports_tree_reference)
    with repository.lock_read():
        for inv, revid in repository._iter_inventories(revids, ordering):
            if inv is None:
                continue
            inv_delta = inv._make_delta(prev_inv)
            lines = serializer.delta_to_lines(prev_inv.revision_id, inv.revision_id, inv_delta)
            yield ChunkedContentFactory(inv.revision_id, None, None, lines, chunks_are_lines=True)
            prev_inv = inv