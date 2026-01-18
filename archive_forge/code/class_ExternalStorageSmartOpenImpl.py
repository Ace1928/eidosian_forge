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
class ExternalStorageSmartOpenImpl(ExternalStorage):
    """The external storage class implemented by smart_open.
    (https://github.com/RaRe-Technologies/smart_open)

    Smart open supports multiple backend with the same APIs.

    To use this implementation, you should pre-create the given uri.
    For example, if your uri is a local file path, you should pre-create
    the directory.

    Args:
        uri: Storage URI used for smart open.
        prefix: Prefix of objects that are stored.
        override_transport_params: Overriding the default value of
            transport_params for smart-open library.

    Raises:
        ModuleNotFoundError: If it fails to setup.
            For example, if smart open library
            is not downloaded, this will fail.
    """

    def __init__(self, uri: str or list, prefix: str=DEFAULT_OBJECT_PREFIX, override_transport_params: dict=None, buffer_size=1024 * 1024):
        try:
            from smart_open import open
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f'Smart open is chosen to be a object spilling external storage, but smart_open and boto3 is not downloaded. Original error: {e}')
        assert uri is not None, 'uri should be provided to use object spilling.'
        if isinstance(uri, str):
            uri = [uri]
        assert isinstance(uri, list), 'uri must be a single string or list of strings.'
        assert isinstance(buffer_size, int), 'buffer_size must be an integer.'
        uri_is_s3 = [u.startswith('s3://') for u in uri]
        self.is_for_s3 = all(uri_is_s3)
        if not self.is_for_s3:
            assert not any(uri_is_s3), "all uri's must be s3 or none can be s3."
            self._uris = uri
        else:
            self._uris = [u.strip('/') for u in uri]
        assert len(self._uris) == len(uri)
        self._current_uri_index = random.randrange(0, len(self._uris))
        self.prefix = prefix
        self.override_transport_params = override_transport_params or {}
        if self.is_for_s3:
            import boto3
            self.s3 = boto3.resource(service_name='s3')
            self.transport_params = {'defer_seek': True, 'resource': self.s3, 'buffer_size': buffer_size}
        else:
            self.transport_params = {}
        self.transport_params.update(self.override_transport_params)

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if len(object_refs) == 0:
            return []
        from smart_open import open
        self._current_uri_index = (self._current_uri_index + 1) % len(self._uris)
        uri = self._uris[self._current_uri_index]
        key = f'{self.prefix}-{_get_unique_spill_filename(object_refs)}'
        url = f'{uri}/{key}'
        with open(url, mode='wb', transport_params=self.transport_params) as file_like:
            return self._write_multiple_objects(file_like, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        from smart_open import open
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with open(base_url, 'rb', transport_params=self.transport_params) as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                owner_address = f.read(address_len)
                total += buf_len
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        pass

    def destroy_external_storage(self):
        pass