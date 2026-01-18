from __future__ import annotations
import asyncio
import random
import inspect
import aiohttpx
import functools
import subprocess
from pydantic import BaseModel
from urllib.parse import urlparse
from lazyops.libs.proxyobj import ProxyObject, proxied
from .base import BaseGlobalClient, cachify
from .utils import aget_root_domain, get_user_agent, http_retry_wrapper
from typing import Optional, Type, TypeVar, Literal, Union, Set, Awaitable, Any, Dict, List, Callable, overload, TYPE_CHECKING
def _fetch_content_type(self, url: str) -> Optional[str]:
    """
        Fetches the content type
        """
    try:
        response = self.api.head(url, follow_redirects=True)
        return response.headers.get('content-type')
    except Exception as e:
        return None