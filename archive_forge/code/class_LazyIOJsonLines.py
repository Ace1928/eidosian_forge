from ._base import *
class LazyIOJsonLines(LazyIOBase):
    newline = '\n'
    num_write_flush = 1000
    serializer = LazyJson

    def __init__(self, filename: str, mode: ModeIO=ModeIO.auto, allow_deletion: bool=False, newline=None, num_write_flush=None, serializer=None, *args, **kwargs):
        self._io_writes = 0
        self._line_map = []
        self.newline = newline or LazyIOJsonLines.newline
        self._num_write_flush = num_write_flush or LazyIOJsonLines.num_write_flush
        self.serializer = serializer or LazyIOJsonLines.serializer
        super(LazyIOJsonLines, self).__init__(filename, mode, allow_deletion, *args, **kwargs)

    def close(self):
        super(LazyIOJsonLines, self).close()
        self._io_writes = 0

    def _build_line_mapping(self):
        self._line_map.append(0)
        while self._io.readline():
            self._line_map.append(self._io.tell())

    def _write(self, data, newline=None, flush=False, dumps_kwargs={}, *args, **kwargs):
        newline = newline or self.newline
        self._io.write(self.serializer.dumps(data, **dumps_kwargs))
        self._io.write(newline)
        self._io_writes += 1
        if flush or self._io_writes % self._num_write_flush == 0:
            self.flush()

    def _iterator(self, ignore_errors=True, loads_kwargs={}, *args, **kwargs):
        for line in self._io:
            try:
                line = self.serializer.loads(line, ignore_errors=ignore_errors, **loads_kwargs)
                if line:
                    yield line
            except StopIteration:
                break
            except Exception as e:
                if not ignore_errors:
                    raise ValueError(str(e))

    def _readlines(self, as_iter=False, as_list=True, ignore_errors=True, loads_kwargs={}, *args, **kwargs):
        if as_iter:
            return self._iterator(ignore_errors=ignore_errors, loads_kwargs=loads_kwargs)
        if as_list:
            return [i for i in self._iterator(ignore_errors=ignore_errors, loads_kwargs={'recursive': True})]
        return self._io.readlines()

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