import json
from typing import Any, List
from urllib import parse
import pathlib
from filelock import FileLock
from ray.workflow.storage.base import Storage
from ray.workflow.storage.filesystem import FilesystemStorageImpl
import ray.cloudpickle
from ray.workflow import serialization_context
class LoggedStorage(FilesystemStorageImpl):
    """A storage records all writing to storage sequentially."""

    def __init__(self, workflow_root_dir: str):
        super().__init__(workflow_root_dir)
        self._log_dir = self._workflow_root_dir
        self._count = self._log_dir / 'count.log'
        self._op_counter = self._log_dir / 'op_counter.pkl'
        if not self._log_dir.exists():
            self._log_dir.mkdir()
        with FileLock(str(self._workflow_root_dir / '.lock')):
            if not self._count.exists():
                with open(self._count, 'x') as f:
                    f.write('0')
            if not self._op_counter.exists():
                with open(self._op_counter, 'wb') as f:
                    ray.cloudpickle.dump({}, f)

    def get_op_counter(self):
        with FileLock(str(self._log_dir / '.lock')):
            with open(self._op_counter, 'rb') as f:
                counter = ray.cloudpickle.load(f)
                return counter

    def update_count(self, op: str, key):
        counter = None
        with open(self._op_counter, 'rb') as f:
            counter = ray.cloudpickle.load(f)
        if op not in counter:
            counter[op] = []
        counter[op].append(key)
        with open(self._op_counter, 'wb') as f:
            ray.cloudpickle.dump(counter, f)

    async def put(self, key: str, data: Any, is_json: bool=False) -> None:
        with FileLock(str(self._log_dir / '.lock')):
            self.update_count('put', key)
            with open(self._count, 'r') as f:
                count = int(f.read())
            k1 = self._log_dir / f'{count}.metadata.json'
            k2 = self._log_dir / f'{count}.value'
            await super().put(str(k1), {'operation': 'put', 'key': key, 'is_json': is_json}, is_json=True)
            await super().put(str(k2), data, is_json=is_json)
            with open(self._count, 'w') as f:
                f.write(str(count + 1))

    async def get(self, key: str, is_json=False) -> None:
        with FileLock(str(self._log_dir / '.lock')):
            self.update_count('get', key)

    async def delete_prefix(self, key: str) -> None:
        with FileLock(str(self._log_dir / '.lock')):
            with open(self._count, 'r') as f:
                count = int(f.read())
            k1 = self._log_dir / f'{count}.metadata.json'
            await super().put(str(k1), {'operation': 'delete_prefix', 'key': key}, is_json=True)
            with open(self._count, 'w') as f:
                f.write(str(count + 1))

    def get_metadata(self, index: int) -> Any:
        with open(self._log_dir / f'{index}.metadata.json') as f:
            return json.load(f)

    def get_value(self, index: int, is_json: bool) -> Any:
        path = self._log_dir / f'{index}.value'
        if is_json:
            with open(path) as f:
                return json.load(f)
        else:
            with open(path, 'rb') as f:
                with serialization_context.workflow_args_keeping_context():
                    return ray.cloudpickle.load(f)

    def __len__(self):
        with open(self._count, 'r') as f:
            return int(f.read())