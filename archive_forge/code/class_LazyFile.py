from ._base import *
class LazyFile:

    def __init__(self, filename, method='tmp', overwrite=True, cleanup=True):
        self._dest_filename = filename
        self._method = method
        self.overwrite = overwrite
        self.cleanup = cleanup
        self._tmp_file = NamedTemporaryFile(delete=False) if method == 'tmp' else BytesIO()

    @property
    def tmpfile(self):
        return self._tmp_file

    @property
    def tmpfile_path(self):
        if self._tmp_file and self._method == 'tmp':
            return self._tmp_file.name
        return None

    @property
    def is_closed(self):
        return bool(self._tmp_file is not None)

    def write(self):
        assert not self.is_closed
        assert File.exists(self._dest_filename) is False or self.overwrite is True
        if self._method == 'tmp':
            data = gfile(self._tmp_file.name, 'rb').read()
        else:
            data = self._tmp_file.getvalue()
        with gfile(self._dest_filename, 'wb') as f:
            f.write(data)
        if self.cleanup and self._method == 'tmp':
            os.remove(self._tmp_file.name)
        self._tmp_file = None