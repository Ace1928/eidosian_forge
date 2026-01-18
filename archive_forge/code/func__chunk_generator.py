import fnmatch
import io
import os
import shutil
import tarfile
from typing import Optional, Tuple, Dict, Generator, Union, List
import ray
from ray.util.annotations import DeveloperAPI
from ray.air._internal.filelock import TempFileLock
from ray.air.util.node import _get_node_id_from_node_ip, _force_on_node
def _chunk_generator(self) -> Generator[bytes, None, None]:
    self.stream.seek(0)
    data = self.stream.read(self.chunk_size)
    while data:
        yield data
        data = self.stream.read(self.chunk_size)