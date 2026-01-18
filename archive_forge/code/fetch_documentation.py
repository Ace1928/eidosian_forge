import posixpath
import stat
from dulwich.object_store import tree_lookup_path
from dulwich.objects import (S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Commit, Tag,
from .. import debug, errors, osutils, trace
from ..bzr.inventory import (InventoryDirectory, InventoryFile, InventoryLink,
from ..bzr.inventorytree import InventoryRevisionTree
from ..bzr.testament import StrictTestament3
from ..bzr.versionedfile import ChunkedContentFactory
from ..errors import BzrError
from ..revision import NULL_REVISION
from ..transport import NoSuchFile
from ..tree import InterTree
from ..tsort import topo_sort
from .mapping import (DEFAULT_FILE_MODE, decode_git_path, mode_is_executable,
from .object_store import LRUTreeCache, _tree_to_objects
Import a set of git objects into a bzr repository.

    :param repo: Target Bazaar repository
    :param mapping: Mapping to use
    :param object_iter: Iterator over Git objects.
    :return: Tuple with pack hints and last imported revision id
    