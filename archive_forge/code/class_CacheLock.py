from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
class CacheLock:

    def __init__(self, fd, key):
        self.fd = fd
        self.key = key

    def save(self, value):
        json_utf8 = encode(value).encode('utf-8')
        try:
            self.fd.write(json_utf8)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()