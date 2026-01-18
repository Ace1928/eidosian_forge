from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
def _prepare_url(self, url: str) -> URL:
    """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
    merge_url = URL(url)
    if merge_url.is_relative_url:
        merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b'/')
        return self.base_url.copy_with(raw_path=merge_raw_path)
    return merge_url