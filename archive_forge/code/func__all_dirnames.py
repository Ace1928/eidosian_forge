from fsspec import AbstractFileSystem
from fsspec.utils import tokenize
def _all_dirnames(self, paths):
    """Returns *all* directory names for each path in paths, including intermediate
        ones.

        Parameters
        ----------
        paths: Iterable of path strings
        """
    if len(paths) == 0:
        return set()
    dirnames = {self._parent(path) for path in paths} - {self.root_marker}
    return dirnames | self._all_dirnames(dirnames)