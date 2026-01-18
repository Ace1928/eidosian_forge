import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from threading import Lock
from typing import Dict, List, Optional, Tuple
class InMemoryStorage(Storage):
    """An in-memory implementation of the Storage interface. This implementation
    is not durable"""

    def __init__(self):
        self._version = 0
        self._tables = defaultdict(dict)
        self._lock = Lock()

    def batch_update(self, table: str, mutation: Dict[str, str]=None, deletion: List[str]=None, expected_version: Optional[int]=None) -> StoreStatus:
        mutation = mutation if mutation else {}
        deletion = deletion if deletion else []
        with self._lock:
            if expected_version is not None and expected_version != self._version:
                return StoreStatus(False, self._version)
            self._version += 1
            key_value_pairs_with_version = {key: VersionedValue(value, self._version) for key, value in mutation.items()}
            self._tables[table].update(key_value_pairs_with_version)
            for deleted_key in deletion:
                self._tables[table].pop(deleted_key, None)
            return StoreStatus(True, self._version)

    def update(self, table: str, key: str, value: str, expected_entry_version: Optional[int]=None, expected_storage_version: Optional[int]=None, insert_only: bool=False) -> StoreStatus:
        with self._lock:
            if expected_storage_version is not None and expected_storage_version != self._version:
                return StoreStatus(False, self._version)
            if insert_only and key in self._tables[table]:
                return StoreStatus(False, self._version)
            _, version = self._tables[table].get(key, (None, -1))
            if expected_entry_version is not None and expected_entry_version != version:
                return StoreStatus(False, self._version)
            self._version += 1
            self._tables[table][key] = VersionedValue(value, self._version)
            return StoreStatus(True, self._version)

    def get_all(self, table: str) -> Tuple[Dict[str, VersionedValue], int]:
        with self._lock:
            return (copy.deepcopy(self._tables[table]), self._version)

    def get(self, table: str, keys: List[str]) -> Tuple[Dict[str, VersionedValue], int]:
        if not keys:
            return self.get_all(table)
        with self._lock:
            result = {}
            for key in keys:
                if key in self._tables.get(table, {}):
                    result[key] = self._tables[table][key]
            return StoreStatus(result, self._version)

    def get_version(self) -> int:
        return self._version