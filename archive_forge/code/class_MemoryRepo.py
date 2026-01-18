import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
class MemoryRepo(BaseRepo):
    """Repo that stores refs, objects, and named files in memory.

    MemoryRepos are always bare: they have no working tree and no index, since
    those have a stronger dependency on the filesystem.
    """

    def __init__(self) -> None:
        from .config import ConfigFile
        self._reflog: List[Any] = []
        refs_container = DictRefsContainer({}, logger=self._append_reflog)
        BaseRepo.__init__(self, MemoryObjectStore(), refs_container)
        self._named_files: Dict[str, bytes] = {}
        self.bare = True
        self._config = ConfigFile()
        self._description = None

    def _append_reflog(self, *args):
        self._reflog.append(args)

    def set_description(self, description):
        self._description = description

    def get_description(self):
        return self._description

    def _determine_file_mode(self):
        """Probe the file-system to determine whether permissions can be trusted.

        Returns: True if permissions can be trusted, False otherwise.
        """
        return sys.platform != 'win32'

    def _determine_symlinks(self):
        """Probe the file-system to determine whether permissions can be trusted.

        Returns: True if permissions can be trusted, False otherwise.
        """
        return sys.platform != 'win32'

    def _put_named_file(self, path, contents):
        """Write a file to the control dir with the given name and contents.

        Args:
          path: The path to the file, relative to the control dir.
          contents: A string to write to the file.
        """
        self._named_files[path] = contents

    def _del_named_file(self, path):
        try:
            del self._named_files[path]
        except KeyError:
            pass

    def get_named_file(self, path, basedir=None):
        """Get a file from the control dir with a specific name.

        Although the filename should be interpreted as a filename relative to
        the control dir in a disk-baked Repo, the object returned need not be
        pointing to a file in that location.

        Args:
          path: The path to the file, relative to the control dir.
        Returns: An open file object, or None if the file does not exist.
        """
        contents = self._named_files.get(path, None)
        if contents is None:
            return None
        return BytesIO(contents)

    def open_index(self):
        """Fail to open index for this repo, since it is bare.

        Raises:
          NoIndexPresent: Raised when no index is present
        """
        raise NoIndexPresent

    def get_config(self):
        """Retrieve the config object.

        Returns: `ConfigFile` object.
        """
        return self._config

    @classmethod
    def init_bare(cls, objects, refs):
        """Create a new bare repository in memory.

        Args:
          objects: Objects for the new repository,
            as iterable
          refs: Refs as dictionary, mapping names
            to object SHA1s
        """
        ret = cls()
        for obj in objects:
            ret.object_store.add_object(obj)
        for refname, sha in refs.items():
            ret.refs.add_if_new(refname, sha)
        ret._init_files(bare=True)
        return ret