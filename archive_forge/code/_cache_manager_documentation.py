import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
Prepare the strategy to delete one or more revisions cached locally.

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
        