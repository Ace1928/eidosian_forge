import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
class TarballContains(Matcher):
    """Matches if the given tarball contains the given paths.

    Uses TarFile.getnames() to get the paths out of the tarball.
    """

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.path_matcher = Equals(sorted(self.paths))

    def match(self, tarball_path):
        f = open(tarball_path, 'rb')
        try:
            tarball = tarfile.open(tarball_path, fileobj=f)
            try:
                return self.path_matcher.match(sorted(tarball.getnames()))
            finally:
                tarball.close()
        finally:
            f.close()