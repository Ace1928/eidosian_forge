import inspect
import io
import os
import re
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import unquote
from huggingface_hub.constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, REPO_TYPES_URL_PREFIXES
from huggingface_hub.utils import get_session
from .utils import (
from .utils.sha import sha256, sha_fileobj
def _upload_parts_iteratively(operation: 'CommitOperationAdd', sorted_parts_urls: List[str], chunk_size: int) -> List[Dict]:
    headers = []
    with operation.as_file(with_tqdm=True) as fileobj:
        for part_idx, part_upload_url in enumerate(sorted_parts_urls):
            with SliceFileObj(fileobj, seek_from=chunk_size * part_idx, read_limit=chunk_size) as fileobj_slice:
                part_upload_res = http_backoff('PUT', part_upload_url, data=fileobj_slice, retry_on_status_codes=(500, 502, 503, 504))
                hf_raise_for_status(part_upload_res)
                headers.append(part_upload_res.headers)
    return headers