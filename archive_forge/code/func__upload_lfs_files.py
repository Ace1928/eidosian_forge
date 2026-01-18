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
def _upload_lfs_files(*, additions: List[CommitOperationAdd], repo_type: str, repo_id: str, token: Optional[str], endpoint: Optional[str]=None, num_threads: int=5, revision: Optional[str]=None):
    """
    Uploads the content of `additions` to the Hub using the large file storage protocol.

    Relevant external documentation:
        - LFS Batch API: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        additions (`List` of `CommitOperationAdd`):
            The files to be uploaded
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )
        num_threads (`int`, *optional*):
            The number of concurrent threads to use when uploading. Defaults to 5.
        revision (`str`, *optional*):
            The git revision to upload to.

    Raises: `RuntimeError` if an upload failed for any reason

    Raises: `ValueError` if the server returns malformed responses

    Raises: `requests.HTTPError` if the LFS batch endpoint returned an HTTP
        error

    """
    batch_actions: List[Dict] = []
    for chunk in chunk_iterable(additions, chunk_size=256):
        batch_actions_chunk, batch_errors_chunk = post_lfs_batch_info(upload_infos=[op.upload_info for op in chunk], token=token, repo_id=repo_id, repo_type=repo_type, revision=revision, endpoint=endpoint)
        if batch_errors_chunk:
            message = '\n'.join([f'Encountered error for file with OID {err.get('oid')}: `{err.get('error', {}).get('message')}' for err in batch_errors_chunk])
            raise ValueError(f'LFS batch endpoint returned errors:\n{message}')
        batch_actions += batch_actions_chunk
    oid2addop = {add_op.upload_info.sha256.hex(): add_op for add_op in additions}
    filtered_actions = []
    for action in batch_actions:
        if action.get('actions') is None:
            logger.debug(f'Content of file {oid2addop[action['oid']].path_in_repo} is already present upstream - skipping upload.')
        else:
            filtered_actions.append(action)
    if len(filtered_actions) == 0:
        logger.debug('No LFS files to upload.')
        return

    def _wrapped_lfs_upload(batch_action) -> None:
        try:
            operation = oid2addop[batch_action['oid']]
            lfs_upload(operation=operation, lfs_batch_action=batch_action, token=token)
        except Exception as exc:
            raise RuntimeError(f"Error while uploading '{operation.path_in_repo}' to the Hub.") from exc
    if HF_HUB_ENABLE_HF_TRANSFER:
        logger.debug(f'Uploading {len(filtered_actions)} LFS files to the Hub using `hf_transfer`.')
        for action in hf_tqdm(filtered_actions):
            _wrapped_lfs_upload(action)
    elif len(filtered_actions) == 1:
        logger.debug('Uploading 1 LFS file to the Hub')
        _wrapped_lfs_upload(filtered_actions[0])
    else:
        logger.debug(f'Uploading {len(filtered_actions)} LFS files to the Hub using up to {num_threads} threads concurrently')
        thread_map(_wrapped_lfs_upload, filtered_actions, desc=f'Upload {len(filtered_actions)} LFS files', max_workers=num_threads, tqdm_class=hf_tqdm)