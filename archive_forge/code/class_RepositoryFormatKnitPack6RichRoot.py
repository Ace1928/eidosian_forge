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
class RepositoryFormatKnitPack6RichRoot(RepositoryFormatPack):
    """A repository with rich roots, no subtrees, stacking and btree indexes.

    1.6-rich-root with B+Tree indices.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    rich_root_data = True
    supports_tree_reference = False
    supports_external_lookups = True
    index_builder_class = btree_index.BTreeBuilder
    index_class = btree_index.BTreeGraphIndex

    @property
    def _serializer(self):
        return xml6.serializer_v6

    def _get_matching_bzrdir(self):
        return controldir.format_registry.make_controldir('1.9-rich-root')

    def _ignore_setting_bzrdir(self, format):
        pass
    _matchingcontroldir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        """See RepositoryFormat.get_format_string()."""
        return b'Bazaar RepositoryFormatKnitPack6RichRoot (bzr 1.9)\n'

    def get_format_description(self):
        return 'Packs 6 rich-root (uses btree indexes, requires bzr 1.9)'