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
def _pack_dir(source_dir: str, exclude: Optional[List]=None, files_stats: Optional[Dict[str, Tuple[float, int]]]=None) -> io.BytesIO:
    """Pack whole directory contents into an uncompressed tarfile.

    This function accepts a ``files_stats`` argument. If given, only files
    whose stats differ from these stats will be packed.

    The main use case for this is that we can collect information about files
    already existing in the target directory, and only pack files that have
    been updated. This is similar to how cloud syncing utilities decide
    which files to transfer.

    Args:
        source_dir: Path to local directory to pack into tarfile.
        exclude: Pattern of files to exclude, e.g.
            ``["*/checkpoint_*]`` to exclude trial checkpoints.
        files_stats: Dict of relative filenames mapping to a tuple of
            (mtime, filesize). Only files that differ from these stats
            will be packed.

    Returns:
        Tarfile as a stream object.
    """

    def _should_exclude(candidate: str) -> bool:
        if not exclude:
            return False
        for excl in exclude:
            if fnmatch.fnmatch(candidate, excl):
                return True
        return False
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as tar:
        if not files_stats and (not exclude):
            tar.add(source_dir, arcname='', recursive=True)
        else:
            files_stats = files_stats or {}
            tar.add(source_dir, arcname='', recursive=False)
            for root, dirs, files in os.walk(source_dir, topdown=False):
                rel_root = os.path.relpath(root, source_dir)
                for dir in dirs:
                    key = os.path.join(rel_root, dir)
                    tar.add(os.path.join(source_dir, key), arcname=key, recursive=False)
                for file in files:
                    key = os.path.join(rel_root, file)
                    stat = os.lstat(os.path.join(source_dir, key))
                    file_stat = (stat.st_mtime, stat.st_size)
                    if _should_exclude(key):
                        continue
                    if key in files_stats and files_stats[key] == file_stat:
                        continue
                    tar.add(os.path.join(source_dir, key), arcname=key)
    return stream