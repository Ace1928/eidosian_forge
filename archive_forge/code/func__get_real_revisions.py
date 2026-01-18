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
def _get_real_revisions(self):
    if self.__real_revisions is None:
        self.__real_revisions = []
        bundle_reader = self.get_bundle_reader()
        for bytes, metadata, repo_kind, revision_id, file_id in bundle_reader.iter_records():
            if repo_kind == 'info':
                serializer = self._serializer.get_source_serializer(metadata)
            if repo_kind == 'revision':
                rev = serializer.read_revision_from_string(bytes)
                self.__real_revisions.append(rev)
    return self.__real_revisions