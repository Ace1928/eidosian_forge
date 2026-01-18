from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
def _do_check_inventories(self, checker, bar):
    """Helper for _check_inventories."""
    revno = 0
    keys = {'chk_bytes': set(), 'inventories': set(), 'texts': set()}
    kinds = ['chk_bytes', 'texts']
    count = len(checker.pending_keys)
    bar.update(gettext('inventories'), 0, 2)
    current_keys = checker.pending_keys
    checker.pending_keys = {}
    for key in current_keys:
        if key[0] != 'inventories' and key[0] not in kinds:
            checker._report_items.append('unknown key type {!r}'.format(key))
        keys[key[0]].add(key[1:])
    if keys['inventories']:
        last_object = None
        for record in self.inventories.check(keys=keys['inventories']):
            if record.storage_kind == 'absent':
                checker._report_items.append('Missing inventory {{{}}}'.format(record.key))
            else:
                last_object = self._check_record('inventories', record, checker, last_object, current_keys[('inventories',) + record.key])
        del keys['inventories']
    else:
        return
    bar.update(gettext('texts'), 1)
    while checker.pending_keys or keys['chk_bytes'] or keys['texts']:
        current_keys = checker.pending_keys
        checker.pending_keys = {}
        for key in current_keys:
            if key[0] not in kinds:
                checker._report_items.append('unknown key type {!r}'.format(key))
            keys[key[0]].add(key[1:])
        for kind in kinds:
            if keys[kind]:
                last_object = None
                for record in getattr(self, kind).check(keys=keys[kind]):
                    if record.storage_kind == 'absent':
                        checker._report_items.append('Missing {} {{{}}}'.format(kind, record.key))
                    else:
                        last_object = self._check_record(kind, record, checker, last_object, current_keys[(kind,) + record.key])
                keys[kind] = set()
                break