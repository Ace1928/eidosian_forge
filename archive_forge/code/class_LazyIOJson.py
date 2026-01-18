from ._base import *
class LazyIOJson(LazyIOBase):
    serializer = LazyJson

    def __init__(self, filename: str, mode: ModeIO=ModeIO.read, allow_deletion: bool=False, serializer=None, *args, **kwargs):
        self.serializer = serializer or LazyIOJson.serializer
        super(LazyIOJson, self).__init__(filename, mode, allow_deletion, *args, **kwargs)

    def _write(self, data, dump_kwargs={}, *args, **kwargs):
        if self.mode.value not in ['w', 'wb']:
            self.setmode(mode=ModeIO.write_binary if 'b' in self.mode.value else ModeIO.write)
        self.serializer.dump(data, self._io, **dump_kwargs)
        self.flush()

    def _read(self, load_kwargs={}, *args, **kwargs):
        return self.serializer.load(self._io, **load_kwargs)