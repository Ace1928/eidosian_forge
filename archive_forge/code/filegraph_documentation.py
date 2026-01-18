import posixpath
import stat
from dulwich.errors import NotTreeError
from dulwich.object_store import tree_lookup_path
from dulwich.objects import SubmoduleEncountered
from ..revision import NULL_REVISION
from .mapping import encode_git_path
File graph access.