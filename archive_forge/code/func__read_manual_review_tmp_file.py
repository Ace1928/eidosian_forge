import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def _read_manual_review_tmp_file(tmp_path: str) -> List[str]:
    """Read the manually reviewed instruction file and return a list of revision hash.

    Example:
        ```txt
        # This is the tmp file content
        ###

        # Commented out line
        123456789 # revision hash

        # Something else
        #      a_newer_hash # 2 days ago
            an_older_hash # 3 days ago
        ```

        ```py
        >>> _read_manual_review_tmp_file(tmp_path)
        ['123456789', 'an_older_hash']
        ```
    """
    with open(tmp_path) as f:
        content = f.read()
    lines = [line.strip() for line in content.split('\n')]
    selected_lines = [line for line in lines if not line.startswith('#')]
    selected_hashes = [line.split('#')[0].strip() for line in selected_lines]
    return [hash for hash in selected_hashes if len(hash) > 0]