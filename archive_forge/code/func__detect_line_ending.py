import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
def _detect_line_ending(content: str) -> Literal['\r', '\n', '\r\n', None]:
    """Detect the line ending of a string. Used by RepoCard to avoid making huge diff on newlines.

    Uses same implementation as in Hub server, keep it in sync.

    Returns:
        str: The detected line ending of the string.
    """
    cr = content.count('\r')
    lf = content.count('\n')
    crlf = content.count('\r\n')
    if cr + lf == 0:
        return None
    if crlf == cr and crlf == lf:
        return '\r\n'
    if cr > lf:
        return '\r'
    else:
        return '\n'