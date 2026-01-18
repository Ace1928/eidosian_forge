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
def _tarball_of_dir(self, dirname, compression, ofile):
    import tarfile
    filename = os.path.basename(ofile.name)
    with tarfile.open(fileobj=ofile, name=filename, mode='w|' + compression) as tarball:
        dirname = dirname.encode(sys.getfilesystemencoding())
        if not dirname.endswith('.bzr'):
            raise ValueError(dirname)
        tarball.add(dirname, '.bzr')