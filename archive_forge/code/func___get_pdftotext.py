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
def __get_pdftotext(self, url: str, validate_url: Optional[bool]=False, raise_errors: Optional[bool]=None, **kwargs) -> Optional[str]:
    """
        Transform a PDF File to Text directly from URL
        """
    if validate_url:
        validate_result = self._validate_url(url)
        if validate_result != True:
            if raise_errors:
                raise ValueError(f'Invalid URL: {url}. {validate_result}')
            self.logger.error(f'Invalid URL: {url}. {validate_result}')
            return None
    cmd = f'curl -s {url} | pdftotext -layout -nopgbrk -eol unix -colspacing 0.7 -y 58 -x 0 -H 741 -W 596 - -'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8')
        return stdout
    except Exception as e:
        stderr = stderr.decode('utf-8')
        self.logger.error(f'Error in pdftotext: {stderr}: {e}')
        if raise_errors:
            raise e
        return None