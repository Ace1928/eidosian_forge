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
def _put_object_to_store(self, metadata, data_size, file_like, object_ref, owner_address):
    worker = ray._private.worker.global_worker
    worker.core_worker.put_file_like_object(metadata, data_size, file_like, object_ref, owner_address)