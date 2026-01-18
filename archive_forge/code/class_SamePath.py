import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
class SamePath(Matcher):
    """Matches if two paths are the same.

    That is, the paths are equal, or they point to the same file but in
    different ways.  The paths do not have to exist.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path

    def match(self, other_path):

        def f(x):
            return os.path.abspath(os.path.realpath(x))
        return Equals(f(self.path)).match(f(other_path))