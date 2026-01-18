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
class ExternalStorage(metaclass=abc.ABCMeta):
    """The base class for external storage.

    This class provides some useful functions for zero-copy object
    put/get from plasma store. Also it specifies the interface for
    object spilling.

    When inheriting this class, please make sure to implement validation
    logic inside __init__ method. When ray instance starts, it will
    instantiating external storage to validate the config.

    Raises:
        ValueError: when given configuration for
            the external storage is invalid.
    """
    HEADER_LENGTH = 24

    def _get_objects_from_store(self, object_refs):
        worker = ray._private.worker.global_worker
        ray_object_pairs = worker.core_worker.get_if_local(object_refs)
        return ray_object_pairs

    def _put_object_to_store(self, metadata, data_size, file_like, object_ref, owner_address):
        worker = ray._private.worker.global_worker
        worker.core_worker.put_file_like_object(metadata, data_size, file_like, object_ref, owner_address)

    def _write_multiple_objects(self, f: IO, object_refs: List[ObjectRef], owner_addresses: List[str], url: str) -> List[str]:
        """Fuse all given objects into a given file handle.

        Args:
            f: File handle to fusion all given object refs.
            object_refs: Object references to fusion to a single file.
            owner_addresses: Owner addresses for the provided objects.
            url: url where the object ref is stored
                in the external storage.

        Return:
            List of urls_with_offset of fused objects.
            The order of returned keys are equivalent to the one
            with given object_refs.
        """
        keys = []
        offset = 0
        ray_object_pairs = self._get_objects_from_store(object_refs)
        for ref, (buf, metadata), owner_address in zip(object_refs, ray_object_pairs, owner_addresses):
            address_len = len(owner_address)
            metadata_len = len(metadata)
            if buf is None and len(metadata) == 0:
                error = f'Object {ref.hex()} does not exist.'
                raise ValueError(error)
            buf_len = 0 if buf is None else len(buf)
            payload = address_len.to_bytes(8, byteorder='little') + metadata_len.to_bytes(8, byteorder='little') + buf_len.to_bytes(8, byteorder='little') + owner_address + metadata + (memoryview(buf) if buf_len else b'')
            payload_len = len(payload)
            assert self.HEADER_LENGTH + address_len + metadata_len + buf_len == payload_len
            written_bytes = f.write(payload)
            assert written_bytes == payload_len
            url_with_offset = create_url_with_offset(url=url, offset=offset, size=written_bytes)
            keys.append(url_with_offset.encode())
            offset += written_bytes
        f.flush()
        return keys

    def _size_check(self, address_len, metadata_len, buffer_len, obtained_data_size):
        """Check whether or not the obtained_data_size is as expected.

        Args:
             metadata_len: Actual metadata length of the object.
             buffer_len: Actual buffer length of the object.
             obtained_data_size: Data size specified in the
                url_with_offset.

        Raises:
            ValueError if obtained_data_size is different from
            address_len + metadata_len + buffer_len +
            24 (first 8 bytes to store length).
        """
        data_size_in_bytes = address_len + metadata_len + buffer_len + self.HEADER_LENGTH
        if data_size_in_bytes != obtained_data_size:
            raise ValueError(f'Obtained data has a size of {data_size_in_bytes}, although it is supposed to have the size of {obtained_data_size}.')

    @abc.abstractmethod
    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        """Spill objects to the external storage. Objects are specified
        by their object refs.

        Args:
            object_refs: The list of the refs of the objects to be spilled.
            owner_addresses: Owner addresses for the provided objects.
        Returns:
            A list of internal URLs with object offset.
        """

    @abc.abstractmethod
    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]) -> int:
        """Restore objects from the external storage.

        Args:
            object_refs: List of object IDs (note that it is not ref).
            url_with_offset_list: List of url_with_offset.

        Returns:
            The total number of bytes restored.
        """

    @abc.abstractmethod
    def delete_spilled_objects(self, urls: List[str]):
        """Delete objects that are spilled to the external storage.

        Args:
            urls: URLs that store spilled object files.

        NOTE: This function should not fail if some of the urls
        do not exist.
        """

    @abc.abstractmethod
    def destroy_external_storage(self):
        """Destroy external storage when a head node is down.

        NOTE: This is currently working when the cluster is
        started by ray.init
        """