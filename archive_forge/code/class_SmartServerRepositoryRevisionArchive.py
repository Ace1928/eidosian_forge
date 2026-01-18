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
class SmartServerRepositoryRevisionArchive(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, revision_id, format, name, root, subdir=None, force_mtime=None):
        """Stream an archive file for a specific revision.
        :param repository: The repository to stream from.
        :param revision_id: Revision for which to export the tree
        :param format: Format (tar, tgz, tbz2, etc)
        :param name: Target file name
        :param root: Name of root directory (or '')
        :param subdir: Subdirectory to export, if not the root
        """
        tree = repository.revision_tree(revision_id)
        if subdir is not None:
            subdir = subdir.decode('utf-8')
        if root is not None:
            root = root.decode('utf-8')
        name = name.decode('utf-8')
        return SuccessfulSmartServerResponse((b'ok',), body_stream=self.body_stream(tree, format.decode('utf-8'), os.path.basename(name), root, subdir, force_mtime))

    def body_stream(self, tree, format, name, root, subdir=None, force_mtime=None):
        with tree.lock_read():
            return tree.archive(format, name, root, subdir, force_mtime)