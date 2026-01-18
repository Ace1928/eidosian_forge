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
def _copy_dir(source_dir: str, target_dir: str, *, exclude: Optional[List]=None, _retry: bool=True) -> None:
    """Copy dir with shutil on the actor."""
    target_dir = os.path.normpath(target_dir)
    try:
        with TempFileLock(f'{target_dir}.lock', timeout=0):
            _delete_path_unsafe(target_dir)
            _ignore = None
            if exclude:

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
            shutil.copytree(source_dir, target_dir, ignore=_ignore)
    except TimeoutError:
        with TempFileLock(f'{target_dir}.lock'):
            pass
        if not os.path.exists(target_dir):
            if _retry:
                _copy_dir(source_dir, target_dir, _retry=False)
            else:
                raise RuntimeError(f"Target directory {target_dir} does not exist and couldn't be recreated. Please raise an issue on GitHub: https://github.com/ray-project/ray/issues")