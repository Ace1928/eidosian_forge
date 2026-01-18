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
class _VersionedFileChecker:

    def __init__(self, repository, text_key_references=None, ancestors=None):
        self.repository = repository
        self.text_index = self.repository._generate_text_key_index(text_key_references=text_key_references, ancestors=ancestors)

    def calculate_file_version_parents(self, text_key):
        """Calculate the correct parents for a file version according to
        the inventories.
        """
        parent_keys = self.text_index[text_key]
        if parent_keys == [_mod_revision.NULL_REVISION]:
            return ()
        return tuple(parent_keys)

    def check_file_version_parents(self, texts, progress_bar=None):
        """Check the parents stored in a versioned file are correct.

        It also detects file versions that are not referenced by their
        corresponding revision's inventory.

        :returns: A tuple of (wrong_parents, dangling_file_versions).
            wrong_parents is a dict mapping {revision_id: (stored_parents,
            correct_parents)} for each revision_id where the stored parents
            are not correct.  dangling_file_versions is a set of (file_id,
            revision_id) tuples for versions that are present in this versioned
            file, but not used by the corresponding inventory.
        """
        local_progress = None
        if progress_bar is None:
            local_progress = ui.ui_factory.nested_progress_bar()
            progress_bar = local_progress
        try:
            return self._check_file_version_parents(texts, progress_bar)
        finally:
            if local_progress:
                local_progress.finished()

    def _check_file_version_parents(self, texts, progress_bar):
        """See check_file_version_parents."""
        wrong_parents = {}
        self.file_ids = {file_id for file_id, _ in self.text_index}
        n_versions = len(self.text_index)
        progress_bar.update(gettext('loading text store'), 0, n_versions)
        parent_map = self.repository.texts.get_parent_map(self.text_index)
        text_keys = self.repository.texts.keys()
        unused_keys = frozenset(text_keys) - set(self.text_index)
        for num, key in enumerate(self.text_index):
            progress_bar.update(gettext('checking text graph'), num, n_versions)
            correct_parents = self.calculate_file_version_parents(key)
            try:
                knit_parents = parent_map[key]
            except errors.RevisionNotPresent:
                knit_parents = None
            if correct_parents != knit_parents:
                wrong_parents[key] = (knit_parents, correct_parents)
        return (wrong_parents, unused_keys)