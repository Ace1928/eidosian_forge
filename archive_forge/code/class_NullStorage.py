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
class NullStorage(ExternalStorage):
    """The class that represents an uninitialized external storage."""

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        raise NotImplementedError('External storage is not initialized')

    def restore_spilled_objects(self, object_refs, url_with_offset_list):
        raise NotImplementedError('External storage is not initialized')

    def delete_spilled_objects(self, urls: List[str]):
        raise NotImplementedError('External storage is not initialized')

    def destroy_external_storage(self):
        raise NotImplementedError('External storage is not initialized')