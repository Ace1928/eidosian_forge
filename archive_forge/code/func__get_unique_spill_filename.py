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
def _get_unique_spill_filename(object_refs: List[ObjectRef]):
    """Generate a unqiue spill file name.

    Args:
        object_refs: objects to be spilled in this file.
    """
    return f'{uuid.uuid4().hex}-multi-{len(object_refs)}'