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
def _install_mp_records(self, versionedfile, records):
    if len(records) == 0:
        return
    d_func = multiparent.MultiParent.from_patch
    vf_records = [(r, m['parents'], m['sha1'], d_func(t)) for r, m, t in records if r not in versionedfile]
    versionedfile.add_mpdiffs(vf_records)