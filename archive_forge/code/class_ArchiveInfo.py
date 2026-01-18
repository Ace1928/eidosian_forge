import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
class ArchiveInfo:
    name = 'archive_info'

    def __init__(self, hash: Optional[str]=None, hashes: Optional[Dict[str, str]]=None) -> None:
        self.hashes = hashes
        self.hash = hash

    @property
    def hash(self) -> Optional[str]:
        return self._hash

    @hash.setter
    def hash(self, value: Optional[str]) -> None:
        if value is not None:
            try:
                hash_name, hash_value = value.split('=', 1)
            except ValueError:
                raise DirectUrlValidationError(f'invalid archive_info.hash format: {value!r}')
            if self.hashes is None:
                self.hashes = {hash_name: hash_value}
            elif hash_name not in self.hashes:
                self.hashes = self.hashes.copy()
                self.hashes[hash_name] = hash_value
        self._hash = value

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['ArchiveInfo']:
        if d is None:
            return None
        return cls(hash=_get(d, str, 'hash'), hashes=_get(d, dict, 'hashes'))

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(hash=self.hash, hashes=self.hashes)