from .. import (
import stat
def _path_to_be_kept(self, path):
    """Does the given path pass the filtering criteria?"""
    if self.excludes and (path in self.excludes or helpers.is_inside_any(self.excludes, path)):
        return False
    if self.includes:
        return path in self.includes or helpers.is_inside_any(self.includes, path)
    return True