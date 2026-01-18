from unittest import TestCase
import os
import tempfile
class FileCreator:
    """A instance which creates a temporary file with a prefix and a given size
    and provides this info to the user.
    Once it gets deleted, it will remove the temporary file as well."""
    __slots__ = ('_size', '_path')

    def __init__(self, size, prefix=''):
        assert size, 'Require size to be larger 0'
        self._path = tempfile.mktemp(prefix=prefix)
        self._size = size
        with open(self._path, 'wb') as fp:
            fp.seek(size - 1)
            fp.write(b'1')
        assert os.path.getsize(self.path) == size

    def __del__(self):
        try:
            os.remove(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    @property
    def path(self):
        return self._path

    @property
    def size(self):
        return self._size