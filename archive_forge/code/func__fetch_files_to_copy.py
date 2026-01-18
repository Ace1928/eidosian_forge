import base64
import io
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
from tqdm.contrib.concurrent import thread_map
from huggingface_hub import get_session
from .constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
from .utils import tqdm as hf_tqdm
@validate_hf_hub_args
def _fetch_files_to_copy(copies: Iterable[CommitOperationCopy], repo_type: str, repo_id: str, token: Optional[str], revision: str, endpoint: Optional[str]=None) -> Dict[Tuple[str, Optional[str]], Union['RepoFile', bytes]]:
    """
    Fetch information about the files to copy.

    For LFS files, we only need their metadata (file size and sha256) while for regular files
    we need to download the raw content from the Hub.

    Args:
        copies (`Iterable` of :class:`CommitOperationCopy`):
            Iterable of :class:`CommitOperationCopy` describing the files to
            copy on the Hub.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.

    Returns: `Dict[Tuple[str, Optional[str]], Union[RepoFile, bytes]]]`
        Key is the file path and revision of the file to copy.
        Value is the raw content as bytes (for regular files) or the file information as a RepoFile (for LFS files).

    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    from .hf_api import HfApi, RepoFolder
    hf_api = HfApi(endpoint=endpoint, token=token)
    files_to_copy: Dict[Tuple[str, Optional[str]], Union['RepoFile', bytes]] = {}
    for src_revision, operations in groupby(copies, key=lambda op: op.src_revision):
        operations = list(operations)
        paths = [op.src_path_in_repo for op in operations]
        for offset in range(0, len(paths), FETCH_LFS_BATCH_SIZE):
            src_repo_files = hf_api.get_paths_info(repo_id=repo_id, paths=paths[offset:offset + FETCH_LFS_BATCH_SIZE], revision=src_revision or revision, repo_type=repo_type)
            for src_repo_file in src_repo_files:
                if isinstance(src_repo_file, RepoFolder):
                    raise NotImplementedError('Copying a folder is not implemented.')
                if src_repo_file.lfs:
                    files_to_copy[src_repo_file.path, src_revision] = src_repo_file
                else:
                    headers = build_hf_headers(token=token)
                    url = hf_hub_url(endpoint=endpoint, repo_type=repo_type, repo_id=repo_id, revision=src_revision or revision, filename=src_repo_file.path)
                    response = get_session().get(url, headers=headers)
                    hf_raise_for_status(response)
                    files_to_copy[src_repo_file.path, src_revision] = response.content
        for operation in operations:
            if (operation.src_path_in_repo, src_revision) not in files_to_copy:
                raise EntryNotFoundError(f'Cannot copy {operation.src_path_in_repo} at revision {src_revision or revision}: file is missing on repo.')
    return files_to_copy