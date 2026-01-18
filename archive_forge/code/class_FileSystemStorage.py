import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
class FileSystemStorage(ExternalStorage):
    """The class for filesystem-like external storage.

    Raises:
        ValueError: Raises directory path to
            spill objects doesn't exist.
    """

    def __init__(self, directory_path, buffer_size=None):
        self._spill_dir_name = DEFAULT_OBJECT_PREFIX
        self._directory_paths = []
        self._current_directory_index = 0
        self._buffer_size = -1
        assert directory_path is not None, 'directory_path should be provided to use object spilling.'
        if isinstance(directory_path, str):
            directory_path = [directory_path]
        assert isinstance(directory_path, list), 'Directory_path must be either a single string or a list of strings'
        if buffer_size is not None:
            assert isinstance(buffer_size, int), 'buffer_size must be an integer.'
            self._buffer_size = buffer_size
        for path in directory_path:
            full_dir_path = os.path.join(path, self._spill_dir_name)
            os.makedirs(full_dir_path, exist_ok=True)
            if not os.path.exists(full_dir_path):
                raise ValueError(f'The given directory path to store objects, {full_dir_path}, could not be created.')
            self._directory_paths.append(full_dir_path)
        assert len(self._directory_paths) == len(directory_path)
        self._current_directory_index = random.randrange(0, len(self._directory_paths))

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if len(object_refs) == 0:
            return []
        self._current_directory_index = (self._current_directory_index + 1) % len(self._directory_paths)
        directory_path = self._directory_paths[self._current_directory_index]
        filename = _get_unique_spill_filename(object_refs)
        url = f'{os.path.join(directory_path, filename)}'
        with open(url, 'wb', buffering=self._buffer_size) as f:
            return self._write_multiple_objects(f, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with open(base_url, 'rb') as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                total += buf_len
                owner_address = f.read(address_len)
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        for url in urls:
            path = parse_url_with_offset(url.decode()).base_url
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def destroy_external_storage(self):
        for directory_path in self._directory_paths:
            self._destroy_external_storage(directory_path)

    def _destroy_external_storage(self, directory_path):
        while os.path.isdir(directory_path):
            try:
                shutil.rmtree(directory_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception('Error cleaning up spill files. You might still have remaining spilled objects inside `ray_spilled_objects` directory.')
                break