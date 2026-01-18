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
def _handle_root(self, target_inv, parent_ids):
    revision_id = target_inv.revision_id
    if self.update_root:
        text_key = (target_inv.root.file_id, revision_id)
        parent_keys = [(target_inv.root.file_id, parent) for parent in parent_ids]
        self._repository.texts.add_lines(text_key, parent_keys, [])
    elif not self._repository.supports_rich_root():
        if target_inv.root.revision != revision_id:
            raise errors.IncompatibleRevision(repr(self._repository))