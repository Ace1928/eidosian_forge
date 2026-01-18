from .. import (
import stat
def _adjust_for_new_root(self, path):
    """Adjust a path given the new root directory of the output."""
    if self.new_root is None:
        return path
    elif path.startswith(self.new_root):
        return path[len(self.new_root):]
    else:
        return path