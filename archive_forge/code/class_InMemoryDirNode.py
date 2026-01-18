import errno
import io
import os
import pathlib
from urllib.parse import urljoin
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.signals import setting_changed
from django.utils._os import safe_join
from django.utils.deconstruct import deconstructible
from django.utils.encoding import filepath_to_uri
from django.utils.functional import cached_property
from django.utils.timezone import now
from .base import Storage
from .mixins import StorageSettingsMixin
class InMemoryDirNode(TimingMixin):
    """
    Helper class representing an in-memory directory node.

    Handle path navigation of directory trees, creating missing nodes if
    needed.
    """

    def __init__(self):
        self._children = {}
        self._initialize_times()

    def resolve(self, path, create_if_missing=False, leaf_cls=None, check_exists=True):
        """
        Navigate current directory tree, returning node matching path or
        creating a new one, if missing.
        - path: path of the node to search
        - create_if_missing: create nodes if not exist. Defaults to False.
        - leaf_cls: expected type of leaf node. Defaults to None.
        - check_exists: if True and the leaf node does not exist, raise a
          FileNotFoundError. Defaults to True.
        """
        path_segments = list(pathlib.Path(path).parts)
        current_node = self
        while path_segments:
            path_segment = path_segments.pop(0)
            if isinstance(current_node, InMemoryFileNode):
                path_segments = os.path.split(path)
                current_path = '/'.join(path_segments[:path_segments.index(path_segment)])
                raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), current_path)
            current_node = current_node._resolve_child(path_segment, create_if_missing, leaf_cls if len(path_segments) == 0 else InMemoryDirNode)
            if current_node is None:
                break
        if current_node is None and check_exists:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        if leaf_cls and (not isinstance(current_node, leaf_cls)):
            error_cls, error_code = (NotADirectoryError, errno.ENOTDIR) if leaf_cls is InMemoryDirNode else (IsADirectoryError, errno.EISDIR)
            raise error_cls(error_code, os.strerror(error_code), path)
        return current_node

    def _resolve_child(self, path_segment, create_if_missing, child_cls):
        if create_if_missing:
            self._update_accessed_time()
            self._update_modified_time()
            return self._children.setdefault(path_segment, child_cls())
        return self._children.get(path_segment)

    def listdir(self):
        directories, files = ([], [])
        for name, entry in self._children.items():
            if isinstance(entry, InMemoryDirNode):
                directories.append(name)
            else:
                files.append(name)
        return (directories, files)

    def remove_child(self, name):
        if name in self._children:
            self._update_accessed_time()
            self._update_modified_time()
            del self._children[name]