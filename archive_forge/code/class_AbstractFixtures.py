import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
class AbstractFixtures(BaseAbstractFixtures):
    """
    Abstract base class containing fixtures that may be overridden in derived
    filesystem-specific classes to run the abstract tests on such filesystems.

    For any particular filesystem some of these fixtures must be overridden,
    such as ``fs`` and ``fs_path``, and others may be overridden if the
    default functions here are not appropriate, such as ``fs_join``.
    """

    @pytest.fixture
    def fs(self):
        raise NotImplementedError('This function must be overridden in derived classes')

    @pytest.fixture
    def fs_join(self):
        """
        Return a function that joins its arguments together into a path.

        Most fsspec implementations join paths in a platform-dependent way,
        but some will override this to always use a forward slash.
        """
        return os.path.join

    @pytest.fixture
    def fs_path(self):
        raise NotImplementedError('This function must be overridden in derived classes')

    @pytest.fixture(scope='class')
    def local_fs(self):
        return LocalFileSystem(auto_mkdir=True)

    @pytest.fixture
    def local_join(self):
        """
        Return a function that joins its arguments together into a path, on
        the local filesystem.
        """
        return os.path.join

    @pytest.fixture
    def local_path(self, tmpdir):
        return tmpdir

    @pytest.fixture
    def supports_empty_directories(self):
        """
        Return whether this implementation supports empty directories.
        """
        return True

    @pytest.fixture
    def fs_sanitize_path(self):
        return lambda x: x