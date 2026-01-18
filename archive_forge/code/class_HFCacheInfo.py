import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
@dataclass(frozen=True)
class HFCacheInfo:
    """Frozen data structure holding information about the entire cache-system.

    This data structure is returned by [`scan_cache_dir`] and is immutable.

    Args:
        size_on_disk (`int`):
            Sum of all valid repo sizes in the cache-system.
        repos (`FrozenSet[CachedRepoInfo]`):
            Set of [`~CachedRepoInfo`] describing all valid cached repos found on the
            cache-system while scanning.
        warnings (`List[CorruptedCacheException]`):
            List of [`~CorruptedCacheException`] that occurred while scanning the cache.
            Those exceptions are captured so that the scan can continue. Corrupted repos
            are skipped from the scan.

    <Tip warning={true}>

    Here `size_on_disk` is equal to the sum of all repo sizes (only blobs). However if
    some cached repos are corrupted, their sizes are not taken into account.

    </Tip>
    """
    size_on_disk: int
    repos: FrozenSet[CachedRepoInfo]
    warnings: List[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of all valid repo sizes in the cache-system as a human-readable
        string.

        Example: "42.2K".
        """
        return _format_size(self.size_on_disk)

    def delete_revisions(self, *revisions: str) -> DeleteCacheStrategy:
        """Prepare the strategy to delete one or more revisions cached locally.

        Input revisions can be any revision hash. If a revision hash is not found in the
        local cache, a warning is thrown but no error is raised. Revisions can be from
        different cached repos since hashes are unique across repos,

        Examples:
        ```py
        >>> from huggingface_hub import scan_cache_dir
        >>> cache_info = scan_cache_dir()
        >>> delete_strategy = cache_info.delete_revisions(
        ...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
        ... )
        >>> print(f"Will free {delete_strategy.expected_freed_size_str}.")
        Will free 7.9K.
        >>> delete_strategy.execute()
        Cache deletion done. Saved 7.9K.
        ```

        ```py
        >>> from huggingface_hub import scan_cache_dir
        >>> scan_cache_dir().delete_revisions(
        ...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa",
        ...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
        ...     "6c0e6080953db56375760c0471a8c5f2929baf11",
        ... ).execute()
        Cache deletion done. Saved 8.6G.
        ```

        <Tip warning={true}>

        `delete_revisions` returns a [`~utils.DeleteCacheStrategy`] object that needs to
        be executed. The [`~utils.DeleteCacheStrategy`] is not meant to be modified but
        allows having a dry run before actually executing the deletion.

        </Tip>
        """
        hashes_to_delete: Set[str] = set(revisions)
        repos_with_revisions: Dict[CachedRepoInfo, Set[CachedRevisionInfo]] = defaultdict(set)
        for repo in self.repos:
            for revision in repo.revisions:
                if revision.commit_hash in hashes_to_delete:
                    repos_with_revisions[repo].add(revision)
                    hashes_to_delete.remove(revision.commit_hash)
        if len(hashes_to_delete) > 0:
            logger.warning(f'Revision(s) not found - cannot delete them: {', '.join(hashes_to_delete)}')
        delete_strategy_blobs: Set[Path] = set()
        delete_strategy_refs: Set[Path] = set()
        delete_strategy_repos: Set[Path] = set()
        delete_strategy_snapshots: Set[Path] = set()
        delete_strategy_expected_freed_size = 0
        for affected_repo, revisions_to_delete in repos_with_revisions.items():
            other_revisions = affected_repo.revisions - revisions_to_delete
            if len(other_revisions) == 0:
                delete_strategy_repos.add(affected_repo.repo_path)
                delete_strategy_expected_freed_size += affected_repo.size_on_disk
                continue
            for revision_to_delete in revisions_to_delete:
                delete_strategy_snapshots.add(revision_to_delete.snapshot_path)
                for ref in revision_to_delete.refs:
                    delete_strategy_refs.add(affected_repo.repo_path / 'refs' / ref)
                for file in revision_to_delete.files:
                    if file.blob_path not in delete_strategy_blobs:
                        is_file_alone = True
                        for revision in other_revisions:
                            for rev_file in revision.files:
                                if file.blob_path == rev_file.blob_path:
                                    is_file_alone = False
                                    break
                            if not is_file_alone:
                                break
                        if is_file_alone:
                            delete_strategy_blobs.add(file.blob_path)
                            delete_strategy_expected_freed_size += file.size_on_disk
        return DeleteCacheStrategy(blobs=frozenset(delete_strategy_blobs), refs=frozenset(delete_strategy_refs), repos=frozenset(delete_strategy_repos), snapshots=frozenset(delete_strategy_snapshots), expected_freed_size=delete_strategy_expected_freed_size)