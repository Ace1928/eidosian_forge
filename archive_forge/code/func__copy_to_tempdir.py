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
def _copy_to_tempdir(self, from_repo):
    tmp_dirname = tempfile.mkdtemp(prefix='tmpbzrclone')
    tmp_bzrdir = from_repo.controldir._format.initialize(tmp_dirname)
    tmp_repo = from_repo._format.initialize(tmp_bzrdir)
    from_repo.copy_content_into(tmp_repo)
    return (tmp_dirname, tmp_repo)