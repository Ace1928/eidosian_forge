import json
from typing import Any, List
from urllib import parse
import pathlib
from filelock import FileLock
from ray.workflow.storage.base import Storage
from ray.workflow.storage.filesystem import FilesystemStorageImpl
import ray.cloudpickle
from ray.workflow import serialization_context
class DebugStorage(Storage):
    """A storage for debugging purpose."""

    def __init__(self, wrapped_storage: 'Storage', path: str):
        self._log_on = True
        self._path = path
        self._wrapped_storage = wrapped_storage
        log_path = pathlib.Path(path)
        parsed = parse.urlparse(wrapped_storage.storage_url)
        log_path = log_path / parsed.scheme.strip('/') / parsed.netloc.strip('/') / parsed.path.strip('/')
        if not log_path.exists():
            log_path.mkdir(parents=True)
        self._logged_storage = LoggedStorage(str(log_path))
        self._op_log_file = log_path / 'debug_operations.log'

    def make_key(self, *names: str) -> str:
        return self._wrapped_storage.make_key(*names)

    async def get(self, key: str, is_json: bool=False) -> Any:
        await self._logged_storage.get(key, is_json)
        return await self._wrapped_storage.get(key, is_json)

    async def put(self, key: str, data: Any, is_json: bool=False) -> None:
        if self._log_on:
            await self._logged_storage.put(key, data, is_json)
        await self._wrapped_storage.put(key, data, is_json)

    async def delete_prefix(self, prefix: str) -> None:
        if self._log_on:
            await self._logged_storage.delete_prefix(prefix)
        await self._wrapped_storage.delete_prefix(prefix)

    async def scan_prefix(self, key_prefix: str) -> List[str]:
        return await self._wrapped_storage.scan_prefix(key_prefix)

    @property
    def storage_url(self) -> str:
        store_url = parse.quote_plus(self._wrapped_storage.storage_url)
        parsed_url = parse.ParseResult(scheme='debug', path=str(pathlib.Path(self._path).absolute()), netloc='', params='', query=f'storage={store_url}', fragment='')
        return parse.urlunparse(parsed_url)

    def __reduce__(self):
        return (DebugStorage, (self._wrapped_storage, self._path))

    @property
    def wrapped_storage(self) -> 'Storage':
        """Get wrapped storage."""
        return self._wrapped_storage

    async def replay(self, index: int) -> None:
        """Replay the a record to the storage.

        Args:
            index: The index of the recorded log to replay.
        """
        log = self.get_log(index)
        op = log['operation']
        if op == 'put':
            is_json = log['is_json']
            data = self.get_value(index, is_json)
            await self._wrapped_storage.put(log['key'], data, is_json)
        elif op == 'delete_prefix':
            await self._wrapped_storage.delete_prefix(log['key'])
        elif op == 'get':
            pass
        else:
            raise ValueError(f"Unknown operation '{op}'.")

    def get_log(self, index: int) -> Any:
        return self._logged_storage.get_metadata(index)

    def get_value(self, index: int, is_json: bool) -> Any:
        return self._logged_storage.get_value(index, is_json)

    def log_off(self):
        self._log_on = False

    def log_on(self):
        self._log_on = True

    def __len__(self):
        return len(self._logged_storage)