from ._base import *
class LazyIOPickle(LazyIOBase):
    serializer = LazyPickler
    num_write_flush = 1000

    def __init__(self, filename: str, mode: ModeIO=ModeIO.readwrite_binary, allow_deletion: bool=False, serializer=None, num_write_flush=None, *args, **kwargs):
        self._line_map = []
        self._io_writes = 0
        self.serializer = serializer or LazyIOPickle.serializer
        self._num_write_flush = num_write_flush or LazyIOPickle.num_write_flush
        super(LazyIOPickle, self).__init__(filename, mode, allow_deletion, *args, **kwargs)

    def _write(self, data, flush=False, dump_kwargs={}, *args, **kwargs):
        if self.mode.value not in ['wb', 'rb+']:
            self.setmode(mode=ModeIO.readwrite_binary)
        self._io.write(self.serializer.dumps(data, **dump_kwargs))
        self._io_writes += 1
        if flush or self._io_writes % self._num_write_flush == 0:
            self.flush()

    def _read(self, load_kwargs={}, *args, **kwargs):
        return self.serializer.load(self._io, **load_kwargs)

    def close(self):
        super(LazyIOPickle, self).close()
        self._io_writes = 0

    def _build_line_mapping(self):
        self._line_map.append(0)
        while self._io.readline():
            self._line_map.append(self._io.tell())

    def __len__(self):
        if not self._line_map:
            self._build_line_mapping()
        return len(self._line_map)

    @timed_cache(120)
    def get_num_lines(self):
        return self.__len__

    @timed_cache(120)
    def __getitem__(self, idx):
        if not self._line_map:
            self._build_line_mapping()
        self._f.seek(self._line_map[idx])
        return self.serializer.loads(self._f.readline(), ignore_errors=True, recursive=True)