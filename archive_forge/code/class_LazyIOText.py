from ._base import *
class LazyIOText(LazyIOBase):
    newline = '\n'
    num_write_flush = 1000

    def __init__(self, filename: str, mode: ModeIO=ModeIO.auto, allow_deletion: bool=False, newline=None, num_write_flush=None, *args, **kwargs):
        self._io_writes = 0
        self.newline = newline or LazyIOText.newline
        self._num_write_flush = num_write_flush or LazyIOText.num_write_flush
        super(LazyIOText, self).__init__(filename, mode, allow_deletion, *args, **kwargs)

    def close(self):
        super(LazyIOText, self).close()
        self._io_writes = 0

    def _write(self, data, newline=None, flush=False, *args, **kwargs):
        newline = newline or self.newline
        self._io.write(data)
        self._io.write(newline)
        self._io_writes += 1
        if flush or self._io_writes % self._num_write_flush == 0:
            self.flush()

    def _readlines(self, as_list=True, strip_newline=True, remove_empty_lines=True, *args, **kwargs):
        if not as_list:
            return self._read()
        texts = self._io.readlines()
        if strip_newline:
            texts = [t.strip() for t in texts]
        if remove_empty_lines:
            texts = [t for t in texts if t]
        return texts