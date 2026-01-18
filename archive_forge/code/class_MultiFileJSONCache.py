from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
class MultiFileJSONCache(MutableMapping):
    writable = True

    def __init__(self, directory):
        self.directory = Path(directory)

    def _filename(self, key):
        return self.directory / f'cache.{key}.json'

    def _glob(self):
        return self.directory.glob('cache.*.json')

    def __iter__(self):
        for path in self._glob():
            cache, key = path.stem.split('.', 1)
            if cache != 'cache':
                continue
            yield key

    def __len__(self):
        return len(list(self._glob()))

    @contextmanager
    def lock(self, key):
        self.directory.mkdir(exist_ok=True, parents=True)
        path = self._filename(key)
        fd = opencew(path)
        try:
            if fd is None:
                yield None
            else:
                yield CacheLock(fd, key)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise Locked(key)
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return read_json(path, always_array=False)
        except FileNotFoundError:
            missing(key)
        except json.decoder.JSONDecodeError:
            return None

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = CombinedJSONCache.dump_cache(self.directory, dict(self))
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache

    def split(self):
        return self

    def filecount(self):
        return len(self)

    def strip_empties(self):
        empties = [key for key, value in self.items() if value is None]
        for key in empties:
            del self[key]
        return len(empties)