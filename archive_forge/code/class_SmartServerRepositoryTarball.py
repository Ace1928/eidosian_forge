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
class SmartServerRepositoryTarball(SmartServerRepositoryRequest):
    """Get the raw repository files as a tarball.

    The returned tarball contains a .bzr control directory which in turn
    contains a repository.

    This takes one parameter, compression, which currently must be
    "", "gz", or "bz2".

    This is used to implement the Repository.copy_content_into operation.
    """

    def do_repository_request(self, repository, compression):
        tmp_dirname, tmp_repo = self._copy_to_tempdir(repository)
        try:
            controldir_name = tmp_dirname + '/.bzr'
            return self._tarfile_response(controldir_name, compression)
        finally:
            osutils.rmtree(tmp_dirname)

    def _copy_to_tempdir(self, from_repo):
        tmp_dirname = tempfile.mkdtemp(prefix='tmpbzrclone')
        tmp_bzrdir = from_repo.controldir._format.initialize(tmp_dirname)
        tmp_repo = from_repo._format.initialize(tmp_bzrdir)
        from_repo.copy_content_into(tmp_repo)
        return (tmp_dirname, tmp_repo)

    def _tarfile_response(self, tmp_dirname, compression):
        with tempfile.NamedTemporaryFile() as temp:
            self._tarball_of_dir(tmp_dirname, compression, temp.file)
            temp.seek(0)
            return SuccessfulSmartServerResponse((b'ok',), temp.read())

    def _tarball_of_dir(self, dirname, compression, ofile):
        import tarfile
        filename = os.path.basename(ofile.name)
        with tarfile.open(fileobj=ofile, name=filename, mode='w|' + compression) as tarball:
            dirname = dirname.encode(sys.getfilesystemencoding())
            if not dirname.endswith('.bzr'):
                raise ValueError(dirname)
            tarball.add(dirname, '.bzr')