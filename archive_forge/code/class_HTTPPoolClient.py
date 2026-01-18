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
class HTTPPoolClient(BaseAPIClient):
    """
    This is a Global HTTP API Client that can be used by any client
    """
    name: Optional[str] = 'http'
    cachify_ttl: Optional[int] = 60 * 60 * 24 * 7
    google_csx_base_url: Optional[str] = 'https://www.googleapis.com/customsearch/v1'
    _pdftotext_enabled: Optional[bool] = None
    _doctotext_enabled: Optional[bool] = None
    _google_csx_api_key: Optional[str] = None
    _google_csx_id: Optional[str] = None

    @property
    def google_csx_api_key(self) -> str:
        """
        Override this to add a Google CSX API Key
        """
        if self._google_csx_api_key is None:
            if hasattr(self.settings, 'clients') and hasattr(self.settings.clients, 'http_pool'):
                self._google_csx_api_key = getattr(self.settings.clients.http_pool, 'google_csx_api_key', None)
        return self._google_csx_api_key

    @property
    def google_csx_id(self) -> str:
        """
        Override this to add a Google CSX ID
        """
        if self._google_csx_id is None:
            if hasattr(self.settings, 'clients') and hasattr(self.settings.clients, 'http_pool'):
                self._google_csx_id = getattr(self.settings.clients.http_pool, 'google_csx_id', None)
        return self._google_csx_id

    @property
    def pdftotext_enabled(self) -> bool:
        """
        Returns whether pdftotext is enabled
        """
        if self._pdftotext_enabled is None:
            try:
                subprocess.check_output(['which', 'pdftotext'])
                self._pdftotext_enabled = True
            except Exception as e:
                self._pdftotext_enabled = False
        return self._pdftotext_enabled

    def cachify_get_name_builder_kwargs(self, func: str, **kwargs) -> Dict[str, Any]:
        """
        Gets the name builder kwargs
        """
        return {'include_http_methods': True, 'special_names': ['pdftotext', 'csx']}

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

    def _get_pdftotext(self, url: str, validate_url: Optional[bool]=False, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, raise_errors: Optional[bool]=None, **kwargs) -> Optional[str]:
        """
        Transform a PDF File to Text directly from URL
        """
        if not self.pdftotext_enabled:
            raise ValueError('pdftotext is not enabled. Please install pdftotext')
        get_func = self.__get_pdftotext
        if retryable:
            get_func = http_retry_wrapper(max_tries=retry_limit + 1)(get_func)
        return get_func(url, raise_errors=raise_errors, **kwargs)

    async def __aget_pdftotext(self, url: str, validate_url: Optional[bool]=False, raise_errors: Optional[bool]=None, **kwargs) -> Optional[str]:
        """
        Transform a PDF File to Text directly from URL
        """
        if validate_url:
            validate_result = await self._avalidate_url(url)
            if validate_result != True:
                if raise_errors:
                    raise ValueError(f'Invalid URL: {url}. {validate_result}')
                self.logger.error(f'Invalid URL: {url}. {validate_result}')
                return None
        cmd = f'curl -s {url} | pdftotext -layout -nopgbrk -eol unix -colspacing 0.7 -y 58 -x 0 -H 741 -W 596 - -'
        process = await asyncio.subprocess.create_subprocess_shell(cmd, shell=True, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await process.communicate()
            stdout = stdout.decode('utf-8')
            return stdout
        except Exception as e:
            stderr = stderr.decode('utf-8')
            self.logger.error(f'Error in pdftotext: {stderr}: {e}')
            if raise_errors:
                raise e
            return None

    async def _aget_pdftotext(self, url: str, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, raise_errors: Optional[bool]=None, **kwargs) -> Optional[str]:
        """
        Transform a PDF File to Text directly from URL
        """
        if not self.pdftotext_enabled:
            raise ValueError('pdftotext is not enabled. Please install pdftotext')
        get_func = self.__aget_pdftotext
        if retryable:
            get_func = http_retry_wrapper(max_tries=retry_limit + 1)(get_func)
        return await get_func(url, raise_errors=raise_errors, **kwargs)

    @cachify.register()
    def get_pdftotext(self, url: str, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> Optional[str]:
        """
        Transform a PDF File to Text directly from URL
        """
        return self._get_pdftotext(url, retryable=retryable, retry_limit=retry_limit, **kwargs)

    @cachify.register()
    async def aget_pdftotext(self, url: str, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> Optional[str]:
        """
        Transform a PDF File to Text directly from URL
        """
        return await self._aget_pdftotext(url, retryable=retryable, retry_limit=retry_limit, **kwargs)
    '\n    Google CSX Methods\n    '

    def get_csx(self, query: str, exact_terms: Optional[str]=None, exclude_terms: Optional[str]=None, file_type: Optional[str]=None, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, **kwargs) -> Dict[str, Union[List[Dict[str, Any]], Any]]:
        """
        Returns the Google CSX Results

        file_type: pdf
        """
        if not self.google_csx_api_key and (not self.google_csx_id):
            raise ValueError('Google CSX API Key and ID are not set')
        params = {'key': self.google_csx_api_key, 'cx': self.google_csx_id, 'q': query}
        if exact_terms:
            params['exactTerms'] = exact_terms
        if exclude_terms:
            params['excludeTerms'] = exclude_terms
        if file_type:
            if 'application/' not in file_type:
                file_type = f'application/{file_type}'
            params['fileType'] = file_type
        if kwargs:
            params.update(kwargs)
        if background:
            return self.pooler.create_background(self.get, url=self.google_csx_base_url, return_type='json', params=params, task_callback=callback, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)
        return self.get(url=self.google_csx_base_url, return_type='json', params=params, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)

    async def aget_csx(self, query: str, exact_terms: Optional[str]=None, exclude_terms: Optional[str]=None, file_type: Optional[str]=None, cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, **kwargs) -> Dict[str, Union[List[Dict[str, Any]], Any]]:
        """
        Returns the Google CSX Results

        file_type: pdf
        """
        if not self.google_csx_api_key and (not self.google_csx_id):
            raise ValueError('Google CSX API Key and ID are not set')
        params = {'key': self.google_csx_api_key, 'cx': self.google_csx_id, 'q': query}
        if exact_terms:
            params['exactTerms'] = exact_terms
        if exclude_terms:
            params['excludeTerms'] = exclude_terms
        if file_type:
            if 'application/' not in file_type:
                file_type = f'application/{file_type}'
            params['fileType'] = file_type
        if kwargs:
            params.update(kwargs)
        if background:
            return self.pooler.create_background(self.aget, url=self.google_csx_base_url, return_type='json', params=params, task_callback=callback, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)
        return await self.aget(url=self.google_csx_base_url, return_type='json', params=params, cachable=cachable, overwrite_cache=overwrite_cache, disable_cache=disable_cache)