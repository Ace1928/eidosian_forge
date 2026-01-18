from ._base import *
def _setup_file(self, filename: str=None):
    if filename is None:
        return
    self._filename = filename
    self._basename = File.base(filename)
    self._directory = File.getdir(filename)
    File.mkdirs(self._directory)
    self._fext = File.ext(self._filename)
    self._ext = ExtIO(self._fext)