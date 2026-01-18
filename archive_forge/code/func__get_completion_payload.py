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
def _get_completion_payload(response_headers: List[Dict], oid: str) -> CompletionPayloadT:
    parts: List[PayloadPartT] = []
    for part_number, header in enumerate(response_headers):
        etag = header.get('etag')
        if etag is None or etag == '':
            raise ValueError(f'Invalid etag (`{etag}`) returned for part {part_number + 1}')
        parts.append({'partNumber': part_number + 1, 'etag': etag})
    return {'oid': oid, 'parts': parts}