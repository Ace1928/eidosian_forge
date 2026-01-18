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
def _fetch_upload_modes(additions: Iterable[CommitOperationAdd], repo_type: str, repo_id: str, token: Optional[str], revision: str, endpoint: Optional[str]=None, create_pr: bool=False, gitignore_content: Optional[str]=None) -> None:
    """
    Requests the Hub "preupload" endpoint to determine whether each input file should be uploaded as a regular git blob
    or as git LFS blob. Input `additions` are mutated in-place with the upload mode.

    Args:
        additions (`Iterable` of :class:`CommitOperationAdd`):
            Iterable of :class:`CommitOperationAdd` describing the files to
            upload to the Hub.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.
        gitignore_content (`str`, *optional*):
            The content of the `.gitignore` file to know which files should be ignored. The order of priority
            is to first check if `gitignore_content` is passed, then check if the `.gitignore` file is present
            in the list of files to commit and finally default to the `.gitignore` file already hosted on the Hub
            (if any).
    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    headers = build_hf_headers(token=token)
    upload_modes: Dict[str, UploadMode] = {}
    should_ignore_info: Dict[str, bool] = {}
    for chunk in chunk_iterable(additions, 256):
        payload: Dict = {'files': [{'path': op.path_in_repo, 'sample': base64.b64encode(op.upload_info.sample).decode('ascii'), 'size': op.upload_info.size, 'sha': op.upload_info.sha256.hex()} for op in chunk]}
        if gitignore_content is not None:
            payload['gitIgnore'] = gitignore_content
        resp = get_session().post(f'{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}', json=payload, headers=headers, params={'create_pr': '1'} if create_pr else None)
        hf_raise_for_status(resp)
        preupload_info = _validate_preupload_info(resp.json())
        upload_modes.update(**{file['path']: file['uploadMode'] for file in preupload_info['files']})
        should_ignore_info.update(**{file['path']: file['shouldIgnore'] for file in preupload_info['files']})
    for addition in additions:
        addition._upload_mode = upload_modes[addition.path_in_repo]
        addition._should_ignore = should_ignore_info[addition.path_in_repo]
    for addition in additions:
        if addition.upload_info.size == 0:
            addition._upload_mode = 'regular'