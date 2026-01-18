from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
class IndexObject(Object):
    """Base for all objects that can be part of the index file, namely Tree, Blob and
    SubModule objects."""
    __slots__ = ('path', 'mode')
    _id_attribute_ = 'path'

    def __init__(self, repo: 'Repo', binsha: bytes, mode: Union[None, int]=None, path: Union[None, PathLike]=None) -> None:
        """Initialize a newly instanced IndexObject.

        :param repo: The :class:`~git.repo.base.Repo` we are located in.
        :param binsha: 20 byte sha1.
        :param mode:
            The stat compatible file mode as int, use the :mod:`stat` module to evaluate
            the information.
        :param path:
            The path to the file in the file system, relative to the git repository
            root, like ``file.ext`` or ``folder/other.ext``.
        :note:
            Path may not be set if the index object has been created directly, as it
            cannot be retrieved without knowing the parent tree.
        """
        super().__init__(repo, binsha)
        if mode is not None:
            self.mode = mode
        if path is not None:
            self.path = path

    def __hash__(self) -> int:
        """
        :return:
            Hash of our path as index items are uniquely identifiable by path, not
            by their data!
        """
        return hash(self.path)

    def _set_cache_(self, attr: str) -> None:
        if attr in IndexObject.__slots__:
            raise AttributeError("Attribute '%s' unset: path and mode attributes must have been set during %s object creation" % (attr, type(self).__name__))
        else:
            super()._set_cache_(attr)

    @property
    def name(self) -> str:
        """:return: Name portion of the path, effectively being the basename"""
        return osp.basename(self.path)

    @property
    def abspath(self) -> PathLike:
        """
        :return:
            Absolute path to this index object in the file system (as opposed to the
            :attr:`path` field which is a path relative to the git repository).

            The returned path will be native to the system and contains '\\' on Windows.
        """
        if self.repo.working_tree_dir is not None:
            return join_path_native(self.repo.working_tree_dir, self.path)
        else:
            raise WorkTreeRepositoryUnsupported('working_tree_dir was None or empty')