import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
def _scan_cached_repo(repo_path: Path) -> CachedRepoInfo:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a [`~CorruptedCacheException`].
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException(f'Repo path is not a directory: {repo_path}')
    if '--' not in repo_path.name:
        raise CorruptedCacheException(f'Repo path is not a valid HuggingFace cache directory: {repo_path}')
    repo_type, repo_id = repo_path.name.split('--', maxsplit=1)
    repo_type = repo_type[:-1]
    repo_id = repo_id.replace('--', '/')
    if repo_type not in {'dataset', 'model', 'space'}:
        raise CorruptedCacheException(f'Repo type must be `dataset`, `model` or `space`, found `{repo_type}` ({repo_path}).')
    blob_stats: Dict[Path, os.stat_result] = {}
    snapshots_path = repo_path / 'snapshots'
    refs_path = repo_path / 'refs'
    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException(f"Snapshots dir doesn't exist in cached repo: {snapshots_path}")
    refs_by_hash: Dict[str, Set[str]] = defaultdict(set)
    if refs_path.exists():
        if refs_path.is_file():
            raise CorruptedCacheException(f'Refs directory cannot be a file: {refs_path}')
        for ref_path in refs_path.glob('**/*'):
            if ref_path.is_dir():
                continue
            ref_name = str(ref_path.relative_to(refs_path))
            with ref_path.open() as f:
                commit_hash = f.read()
            refs_by_hash[commit_hash].add(ref_name)
    cached_revisions: Set[CachedRevisionInfo] = set()
    for revision_path in snapshots_path.iterdir():
        if revision_path.is_file():
            raise CorruptedCacheException(f'Snapshots folder corrupted. Found a file: {revision_path}')
        cached_files = set()
        for file_path in revision_path.glob('**/*'):
            if file_path.is_dir():
                continue
            blob_path = Path(file_path).resolve()
            if not blob_path.exists():
                raise CorruptedCacheException(f'Blob missing (broken symlink): {blob_path}')
            if blob_path not in blob_stats:
                blob_stats[blob_path] = blob_path.stat()
            cached_files.add(CachedFileInfo(file_name=file_path.name, file_path=file_path, size_on_disk=blob_stats[blob_path].st_size, blob_path=blob_path, blob_last_accessed=blob_stats[blob_path].st_atime, blob_last_modified=blob_stats[blob_path].st_mtime))
        if len(cached_files) > 0:
            revision_last_modified = max((blob_stats[file.blob_path].st_mtime for file in cached_files))
        else:
            revision_last_modified = revision_path.stat().st_mtime
        cached_revisions.add(CachedRevisionInfo(commit_hash=revision_path.name, files=frozenset(cached_files), refs=frozenset(refs_by_hash.pop(revision_path.name, set())), size_on_disk=sum((blob_stats[blob_path].st_size for blob_path in set((file.blob_path for file in cached_files)))), snapshot_path=revision_path, last_modified=revision_last_modified))
    if len(refs_by_hash) > 0:
        raise CorruptedCacheException(f'Reference(s) refer to missing commit hashes: {dict(refs_by_hash)} ({repo_path}).')
    if len(blob_stats) > 0:
        repo_last_accessed = max((stat.st_atime for stat in blob_stats.values()))
        repo_last_modified = max((stat.st_mtime for stat in blob_stats.values()))
    else:
        repo_stats = repo_path.stat()
        repo_last_accessed = repo_stats.st_atime
        repo_last_modified = repo_stats.st_mtime
    return CachedRepoInfo(nb_files=len(blob_stats), repo_id=repo_id, repo_path=repo_path, repo_type=repo_type, revisions=frozenset(cached_revisions), size_on_disk=sum((stat.st_size for stat in blob_stats.values())), last_accessed=repo_last_accessed, last_modified=repo_last_modified)