import sys
from fixtures import Fixture
class PythonPathEntry(Fixture):
    """Add a path to sys.path.
    
    If the path is already in sys.path, sys.path will not be altered.
    """

    def __init__(self, directory):
        """Create a PythonPathEntry.

        :param directory: The directory to add to sys.path.
        """
        self.directory = directory

    def _setUp(self):
        if self.directory in sys.path:
            return
        self.addCleanup(sys.path.remove, self.directory)
        sys.path.append(self.directory)