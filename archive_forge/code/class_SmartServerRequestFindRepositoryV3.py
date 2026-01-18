import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestFindRepositoryV3(SmartServerRequestFindRepository):

    def do(self, path):
        """try to find a repository from path upwards

        This operates precisely like 'bzrdir.find_repository'.

        If a bzrdir is not present, an exception is propogated
        rather than 'no branch' because these are different conditions.

        This is the third edition of this method introduced in bzr 1.13, which
        returns information about the network name of the repository format.

        :return: norepository or ok, relpath, rich_root, tree_ref,
            external_lookup, network_name.
        """
        try:
            path, rich_root, tree_ref, external_lookup, name = self._find(path)
            return SuccessfulSmartServerResponse((b'ok', path.encode('utf-8'), rich_root, tree_ref, external_lookup, name))
        except errors.NoRepositoryPresent:
            return FailedSmartServerResponse((b'norepository',))