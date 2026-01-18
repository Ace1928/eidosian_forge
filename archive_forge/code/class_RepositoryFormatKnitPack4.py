from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
class RepositoryFormatKnitPack4(RepositoryFormatPack):
    """A rich-root, no subtrees parameterized Pack repository.

    This repository format uses the xml6 serializer to get:
     - support for recording full info about the tree root

    This format was introduced in 1.0.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    rich_root_data = True
    supports_tree_reference = False

    @property
    def _serializer(self):
        return xml6.serializer_v6
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('rich-root-pack')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar pack repository format 1 with rich root (needs bzr 1.0)\n'

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Packs containing knits with rich root support\n'