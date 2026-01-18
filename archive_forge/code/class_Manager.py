import contextlib
from dataclasses import dataclass
import logging
import os
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.workflow import common, workflow_storage
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from collections import ChainMap
import io
@ray.remote(num_cpus=0)
class Manager:
    """
    Responsible for deduping the serialization/upload of object references.
    """

    def __init__(self):
        self._uploads: Dict[ray.ObjectRef, Upload] = {}
        self._num_uploads = 0

    def ping(self) -> None:
        """
        Trivial function to ensure actor creation is successful.
        """
        return None

    async def save_objectref(self, ref_tuple: Tuple[ray.ObjectRef], workflow_id: 'str') -> Tuple[List[str], ray.ObjectRef]:
        """Serialize and upload an object reference exactly once.

        Args:
            ref_tuple: A 1-element tuple which wraps the reference.

        Returns:
            A pair. The first element is the paths the ref will be uploaded to.
            The second is an object reference to the upload task.
        """
        ref, = ref_tuple
        key = (ref.hex(), workflow_id)
        if key not in self._uploads:
            identifier_ref = common.calculate_identifier.remote(ref)
            upload_task = _put_helper.remote(identifier_ref, ref, workflow_id)
            self._uploads[key] = Upload(identifier_ref=identifier_ref, upload_task=upload_task)
            self._num_uploads += 1
        info = self._uploads[key]
        identifer = await info.identifier_ref
        key = _obj_id_to_key(identifer)
        return (key, info.upload_task)

    async def export_stats(self) -> Dict[str, Any]:
        return {'num_uploads': self._num_uploads}