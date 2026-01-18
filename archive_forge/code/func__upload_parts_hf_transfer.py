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
def _upload_parts_hf_transfer(operation: 'CommitOperationAdd', sorted_parts_urls: List[str], chunk_size: int) -> List[Dict]:
    try:
        from hf_transfer import multipart_upload
    except ImportError:
        raise ValueError("Fast uploading using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available in your environment. Try `pip install hf_transfer`.")
    supports_callback = 'callback' in inspect.signature(multipart_upload).parameters
    if not supports_callback:
        warnings.warn('You are using an outdated version of `hf_transfer`. Consider upgrading to latest version to enable progress bars using `pip install -U hf_transfer`.')
    total = operation.upload_info.size
    desc = operation.path_in_repo
    if len(desc) > 40:
        desc = f'(…){desc[-40:]}'
    disable = True if logger.getEffectiveLevel() == logging.NOTSET else None
    with tqdm(unit='B', unit_scale=True, total=total, initial=0, desc=desc, disable=disable) as progress:
        try:
            output = multipart_upload(file_path=operation.path_or_fileobj, parts_urls=sorted_parts_urls, chunk_size=chunk_size, max_files=128, parallel_failures=127, max_retries=5, **{'callback': progress.update} if supports_callback else {})
        except Exception as e:
            raise RuntimeError('An error occurred while uploading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.') from e
        if not supports_callback:
            progress.update(total)
        return output