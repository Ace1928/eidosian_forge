from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
class FileDBBase:
    """Provides basic facilities to retrieve files of interest, including
    caching facilities to help mapping hexsha's to objects"""

    def __init__(self, root_path):
        """Initialize this instance to look for its files at the given root path
        All subsequent operations will be relative to this path
        :raise InvalidDBRoot:
        **Note:** The base will not perform any accessablity checking as the base
            might not yet be accessible, but become accessible before the first
            access."""
        super().__init__()
        self._root_path = root_path

    def root_path(self):
        """:return: path at which this db operates"""
        return self._root_path

    def db_path(self, rela_path):
        """
        :return: the given relative path relative to our database root, allowing
            to pontentially access datafiles"""
        return join(self._root_path, force_text(rela_path))