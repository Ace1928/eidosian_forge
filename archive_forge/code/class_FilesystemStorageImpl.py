import os
import contextlib
import json
import shutil
import pathlib
from typing import Any, List
import uuid
from ray.workflow.storage.base import Storage, KeyNotFoundError
import ray.cloudpickle
class FilesystemStorageImpl(Storage):
    """Filesystem implementation for accessing workflow storage.

    We do not repeat the same comments for abstract methods in the base class.
    """

    def __init__(self, workflow_root_dir: str):
        self._workflow_root_dir = pathlib.Path(workflow_root_dir)
        if self._workflow_root_dir.exists():
            if not self._workflow_root_dir.is_dir():
                raise ValueError(f'storage path {workflow_root_dir} must be a directory.')
        else:
            self._workflow_root_dir.mkdir(parents=True)

    def make_key(self, *names: str) -> str:
        return os.path.join(str(self._workflow_root_dir), *names)

    async def put(self, key: str, data: Any, is_json: bool=False) -> None:
        if is_json:
            with _open_atomic(pathlib.Path(key), 'w') as f:
                return json.dump(data, f)
        else:
            with _open_atomic(pathlib.Path(key), 'wb') as f:
                return ray.cloudpickle.dump(data, f)

    async def get(self, key: str, is_json: bool=False) -> Any:
        if is_json:
            with _open_atomic(pathlib.Path(key)) as f:
                return json.load(f)
        else:
            with _open_atomic(pathlib.Path(key), 'rb') as f:
                return ray.cloudpickle.load(f)

    async def delete_prefix(self, key_prefix: str) -> None:
        path = pathlib.Path(key_prefix)
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()

    async def scan_prefix(self, key_prefix: str) -> List[str]:
        try:
            path = pathlib.Path(key_prefix)
            return [p.name for p in path.iterdir()]
        except FileNotFoundError:
            return []

    @property
    def storage_url(self) -> str:
        return 'file://' + str(self._workflow_root_dir.absolute())

    def __reduce__(self):
        return (FilesystemStorageImpl, (self._workflow_root_dir,))