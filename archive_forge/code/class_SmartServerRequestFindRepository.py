import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestFindRepository(SmartServerRequestBzrDir):

    def _find(self, path):
        """try to find a repository from path upwards

        This operates precisely like 'bzrdir.find_repository'.

        :return: (relpath, rich_root, tree_ref, external_lookup, network_name).
            All are strings, relpath is a / prefixed path, the next three are
            either 'yes' or 'no', and the last is a repository format network
            name.
        :raises errors.NoRepositoryPresent: When there is no repository
            present.
        """
        bzrdir = BzrDir.open_from_transport(self.transport_from_client_path(path))
        repository = bzrdir.find_repository()
        path = self._repo_relpath(bzrdir.root_transport, repository)
        rich_root, tree_ref, external_lookup = self._format_to_capabilities(repository._format)
        network_name = repository._format.network_name()
        return (path, rich_root, tree_ref, external_lookup, network_name)