import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
class FileCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, 'lock')
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            self.cache_dir = os.getenv('TRITON_CACHE_DIR', '').strip() or default_cache_dir()
            if self.cache_dir:
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, 'lock')
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError('Could not create or locate cache dir')

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError('Could not create or locate cache dir')
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f'__grp__{filename}'
        if not self.has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get('child_paths', None)
        if child_paths is None:
            return None
        result = {}
        for c in child_paths:
            p = self._make_path(c)
            if os.path.exists(p):
                result[c] = p
        return result

    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError('Could not create or locate cache dir')
        grp_contents = json.dumps({'child_paths': sorted(list(group.keys()))})
        grp_filename = f'__grp__{filename}'
        return self.put(grp_contents, grp_filename, binary=False)

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError('Could not create or locate cache dir')
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        rnd_id = random.randint(0, 1000000)
        pid = os.getpid()
        temp_path = f'{filepath}.tmp.pid_{pid}_{rnd_id}'
        mode = 'wb' if binary else 'w'
        with open(temp_path, mode) as f:
            f.write(data)
        os.replace(temp_path, filepath)
        return filepath