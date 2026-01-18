import os
import tempfile
import fixtures
class SecureTempFile(fixtures.Fixture):
    """A fixture for creating a secure temp file."""

    def setUp(self):
        super(SecureTempFile, self).setUp()
        _fd, self.file_name = tempfile.mkstemp()
        os.close(_fd)
        self.addCleanup(os.remove, self.file_name)