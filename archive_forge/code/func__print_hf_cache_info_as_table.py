import time
from argparse import Namespace, _SubParsersAction
from typing import Optional
from ..utils import CacheNotFound, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI, tabulate
def _print_hf_cache_info_as_table(self, hf_cache_info: HFCacheInfo) -> None:
    if self.verbosity == 0:
        print(tabulate(rows=[[repo.repo_id, repo.repo_type, '{:>12}'.format(repo.size_on_disk_str), repo.nb_files, repo.last_accessed_str, repo.last_modified_str, ', '.join(sorted(repo.refs)), str(repo.repo_path)] for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)], headers=['REPO ID', 'REPO TYPE', 'SIZE ON DISK', 'NB FILES', 'LAST_ACCESSED', 'LAST_MODIFIED', 'REFS', 'LOCAL PATH']))
    else:
        print(tabulate(rows=[[repo.repo_id, repo.repo_type, revision.commit_hash, '{:>12}'.format(revision.size_on_disk_str), revision.nb_files, revision.last_modified_str, ', '.join(sorted(revision.refs)), str(revision.snapshot_path)] for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path) for revision in sorted(repo.revisions, key=lambda revision: revision.commit_hash)], headers=['REPO ID', 'REPO TYPE', 'REVISION', 'SIZE ON DISK', 'NB FILES', 'LAST_MODIFIED', 'REFS', 'LOCAL PATH']))