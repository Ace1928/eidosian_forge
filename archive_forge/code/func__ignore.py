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
def _ignore(path, names):
    ignored_names = set()
    rel_path = os.path.relpath(path, source_dir)
    for name in names:
        candidate = os.path.join(rel_path, name)
        for excl in exclude:
            if fnmatch.fnmatch(candidate, excl):
                ignored_names.add(name)
                break
    return ignored_names