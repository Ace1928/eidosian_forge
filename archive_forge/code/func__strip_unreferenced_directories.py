from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
@staticmethod
def _strip_unreferenced_directories(entries):
    """Strip all directories that don't (in)directly contain any files.

        :param entries: List of path strings or (path, previous_path) tuples to process
        """
    directory_used = set()
    directories = []
    for path, previous_path in entries:
        if not path or path[-1] == '/':
            directories.append((path, previous_path))
        else:
            for direntry in directories:
                if osutils.is_inside(direntry[0], path):
                    directory_used.add(direntry[0])
    for path, previous_path in entries:
        if not path.endswith('/') or path in directory_used:
            yield (path, previous_path)