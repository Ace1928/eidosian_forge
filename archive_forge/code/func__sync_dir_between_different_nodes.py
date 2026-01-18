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
def _sync_dir_between_different_nodes(source_ip: str, source_path: str, target_ip: str, target_path: str, force_all: bool=False, exclude: Optional[List]=None, chunk_size_bytes: int=_DEFAULT_CHUNK_SIZE_BYTES, max_size_bytes: Optional[int]=_DEFAULT_MAX_SIZE_BYTES, return_futures: bool=False) -> Union[None, Tuple[ray.ObjectRef, ray.ActorID, ray.ObjectRef]]:
    """Synchronize directory on source node to directory on target node.

    Per default, this function will collect information about already existing
    files in the target directory. Only files that differ in either mtime or
    filesize will be transferred, unless ``force_all=True``.

    Args:
        source_ip: IP of source node.
        source_path: Path to directory on source node.
        target_ip: IP of target node.
        target_path: Path to directory on target node.
        force_all: If True, all files will be transferred (not just differing files).
        exclude: Pattern of files to exclude, e.g.
            ``["*/checkpoint_*]`` to exclude trial checkpoints.
        chunk_size_bytes: Chunk size for data transfer.
        max_size_bytes: If packed data exceeds this value, raise an error before
            transfer. If ``None``, no limit is enforced.
        return_futures: If True, returns a tuple of the unpack future,
            the pack actor, and the files_stats future. If False (default) will
            block until synchronization finished and return None.

    Returns:
        None, or Tuple of unpack future, pack actor, and files_stats future.

    """
    source_node_id = _get_node_id_from_node_ip(source_ip)
    target_node_id = _get_node_id_from_node_ip(target_ip)
    pack_actor_on_source_node = _PackActor.options(num_cpus=0, **_force_on_node(source_node_id))
    unpack_on_target_node = _unpack_from_actor.options(num_cpus=0, **_force_on_node(target_node_id))
    if force_all:
        files_stats = None
    else:
        files_stats = _remote_get_recursive_files_and_stats.options(num_cpus=0, **_force_on_node(target_node_id)).remote(target_path)
    pack_actor = pack_actor_on_source_node.remote(source_dir=source_path, files_stats=files_stats, chunk_size_bytes=chunk_size_bytes, max_size_bytes=max_size_bytes, exclude=exclude)
    unpack_future = unpack_on_target_node.remote(pack_actor, target_path)
    if return_futures:
        return (unpack_future, pack_actor, files_stats)
    return ray.get(unpack_future)