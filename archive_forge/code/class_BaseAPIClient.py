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
@cachify.register_object()
class BaseAPIClient(BaseGlobalClient):
    name: Optional[str] = 'api'
    verbose_errors: Optional[bool] = True
    http_timeout: Optional[float] = None
    _endpoint: Optional[str] = None
    _headers: Optional[Dict[str, Any]] = None
    _api: Optional[aiohttpx.Client] = None

    @property
    def endpoint(self) -> str:
        """
        Returns the endpoint

        - Override this to add custom endpoints
        """
        return self._endpoint

    def get_user_agent(self, *args, **kwargs) -> str:
        """
        Returns the user agent
        """
        return get_user_agent(*args, **kwargs)

    @property
    def headers(self) -> Dict[str, Any]:
        """
        Returns the headers

        - Override this to add custom headers
        """
        return self._headers or {}

    @property
    def api_client_kwargs(self) -> Dict[str, Any]:
        """
        Returns the api client kwargs

        - Override this to add custom kwargs
        """
        return {}

    def cachify_get_exclude_keys(self, func: str, **kwargs) -> List[str]:
        """
        Gets the exclude keys
        """
        return ['background', 'callback', 'retryable', 'retry_limit', 'validate_url', 'cachable', 'disable_cache', 'overwrite_cache']

    def configure_api_client(self, *args, **kwargs) -> aiohttpx.Client:
        """
        Configures the API Client
        """
        if hasattr(self.settings, 'clients') and hasattr(self.settings.clients, 'http_pool'):
            limits = aiohttpx.Limits(max_connections=self.settings.clients.http_pool.max_connections, max_keepalive_connections=self.settings.clients.http_pool.max_keepalive_connections, keepalive_expiry=self.settings.clients.http_pool.keepalive_expiry)
            timeout = self.http_timeout or self.settings.clients.http_pool.default_timeout
        else:
            limits = aiohttpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=60)
            timeout = self.http_timeout or 60
        return aiohttpx.Client(base_url=self.endpoint, limits=limits, timeout=timeout, headers=self.headers, verify=False, **self.api_client_kwargs)

    @property
    def api(self) -> aiohttpx.Client:
        """
        Returns the API Client
        """
        if self._api is None:
            self._api = self.configure_api_client()
        return self._api

    async def areset_api(self):
        """
        Resets the api client
        """
        if self._api:
            await self._api.aclose()
            self._api = None

    def wrap_retryable_method(self, func: Callable[..., Any], retry_limit: Optional[int]=3) -> Union[Callable[..., aiohttpx.Response], Callable[..., Awaitable[aiohttpx.Response]]]:
        """
        Wraps a retryable method
        """
        if isinstance(func, str):
            func = getattr(self.api, func)
        return http_retry_wrapper(max_tries=retry_limit + 1)(func)

    def cachify_validator_is_disabled(self, *args, disable_cache: Optional[bool]=None, **kwargs) -> bool:
        """
        Checks if the function is disabled
        """
        return disable_cache

    def cachify_validator_is_overwrite(self, *args, overwrite_cache: Optional[bool]=None, **kwargs) -> bool:
        """
        Checks if the function is overwrite
        """
        return overwrite_cache

    def cachify_get_name_builder_kwargs(self, func: str, **kwargs) -> Dict[str, Any]:
        """
        Gets the name builder kwargs
        """
        return {'include_http_methods': True}
    '\n    Modified HTTP Methods\n    '

    @overload
    def get(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        ...

    @cachify.register()
    def get(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a GET Request and returns a `ReturnTypeT`
        """
        if background:
            return self.pooler.create_background(self._get, *args, url=url, return_type=return_type, task_callback=callback, retryable=retryable, retry_limit=retry_limit, **kwargs)
        return self._get(*args, url=url, return_type=return_type, retryable=retryable, retry_limit=retry_limit, **kwargs)

    @cachify.register()
    def post(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a POST Request and returns a `ReturnTypeT`
        """
        if background:
            return self.pooler.create_background(self._post, *args, url=url, return_type=return_type, task_callback=callback, retryable=retryable, retry_limit=retry_limit, **kwargs)
        return self._post(*args, url=url, return_type=return_type, retryable=retryable, retry_limit=retry_limit, **kwargs)

    @overload
    async def aget(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a GET Request and returns a `ReturnTypeT`
        """

    @cachify.register()
    async def aget(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a GET Request and returns a `ReturnTypeT`
        """
        if background:
            return self.pooler.create_background(self._aget, *args, url=url, return_type=return_type, task_callback=callback, retryable=retryable, retry_limit=retry_limit, **kwargs)
        return await self._aget(*args, url=url, return_type=return_type, retryable=retryable, retry_limit=retry_limit, **kwargs)

    @cachify.register()
    async def apost(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', cachable: Optional[bool]=True, overwrite_cache: Optional[bool]=None, disable_cache: Optional[bool]=None, background: Optional[bool]=False, callback: Optional[Callable[..., Any]]=None, retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a POST Request and returns a `ReturnTypeT`
        """
        if background:
            return self.pooler.create_background(self._apost, *args, url=url, return_type=return_type, task_callback=callback, retryable=retryable, retry_limit=retry_limit, **kwargs)
        return await self._apost(*args, url=url, return_type=return_type, retryable=retryable, retry_limit=retry_limit, **kwargs)
    '\n    Basic HTTP Methods\n    '

    def _get_(self, url: str, **kwargs) -> aiohttpx.Response:
        """
        Makes a GET request
        """
        return self.api.get(url, **kwargs)

    def _post_(self, url: str, **kwargs) -> aiohttpx.Response:
        """
        Makes a POST request
        """
        return self.api.post(url, **kwargs)

    def _validate_url(self, url: str) -> Union[bool, str]:
        """
        Quickly validates a URL
        """
        try:
            response = self.api.head(url, follow_redirects=True)
            try:
                response.raise_for_status()
                return True
            except Exception as e:
                return f'[{response.status_code}]: {response.text}'
        except Exception as e:
            return f'[{type(e)}]: {str(e)}'

    def _fetch_content_type(self, url: str) -> Optional[str]:
        """
        Fetches the content type
        """
        try:
            response = self.api.head(url, follow_redirects=True)
            return response.headers.get('content-type')
        except Exception as e:
            return None

    async def _aget_(self, url: str, **kwargs) -> aiohttpx.Response:
        """
        Makes a GET request
        """
        return await self.api.async_get(url, **kwargs)

    async def _apost_(self, url: str, **kwargs) -> aiohttpx.Response:
        """
        Makes a POST request
        """
        return await self.api.async_post(url, **kwargs)

    async def _avalidate_url(self, url: str) -> Union[bool, str]:
        """
        Quickly validates a URL
        """
        try:
            response = await self.api.async_head(url, follow_redirects=True)
            try:
                response.raise_for_status()
                return True
            except Exception as e:
                return f'[{response.status_code}] {response.text}'
        except Exception as e:
            return f'[{type(e)}]: {str(e)}'

    async def _afetch_content_type(self, url: str) -> Optional[str]:
        """
        Fetches the content type
        """
        try:
            response = await self.api.async_head(url, follow_redirects=True)
            return response.headers.get('content-type')
        except Exception as e:
            return None
    '\n    Enhanced HTTP Methods\n    '

    def _get(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a GET Request and returns a `ReturnTypeT`
        """
        get_func = self._get_
        if retryable:
            get_func = http_retry_wrapper(max_tries=retry_limit + 1)(get_func)
        response = get_func(url, *args, **kwargs)
        return self.handle_response(response, return_type=return_type)

    def _post(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a POST Request and returns a `ReturnTypeT`
        """
        post_func = self._post_
        if retryable:
            post_func = http_retry_wrapper(max_tries=retry_limit + 1)(post_func)
        response = post_func(url, *args, **kwargs)
        return self.handle_response(response, return_type=return_type)

    async def _aget(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a GET Request and returns a `ReturnTypeT`
        """
        get_func = self._aget_
        if retryable:
            get_func = http_retry_wrapper(max_tries=retry_limit + 1)(get_func)
        response = await get_func(url, *args, **kwargs)
        return self.handle_response(response, return_type=return_type)

    async def _apost(self, url: str, *args, return_type: Optional[Union[ReturnType, ReturnModelT]]='json', retryable: Optional[bool]=False, retry_limit: Optional[int]=3, **kwargs) -> ReturnTypeT:
        """
        Creates a POST Request and returns a `ReturnTypeT`
        """
        post_func = self._apost_
        if retryable:
            post_func = http_retry_wrapper(max_tries=retry_limit + 1)(post_func)
        response = await post_func(url, *args, **kwargs)
        return self.handle_response(response, return_type=return_type)

    def handle_response(self, response: aiohttpx.Response, return_type: Optional[Union[ReturnType, ReturnModelT]]='json') -> Union[ReturnModelT, Dict[str, Any]]:
        """
        Handle the response
        """
        try:
            response.raise_for_status()
            if not return_type:
                return response
            if not isinstance(return_type, str):
                return return_type.model_validate(response.json())
            return_type = str(return_type).lower()
            if return_type == 'json':
                return response.json()
            if return_type == 'text':
                return response.text
            if return_type == 'bytes':
                return response.content
            else:
                raise ValueError(f'Invalid return type: {return_type}')
        except aiohttpx.HTTPStatusError as e:
            if self.settings.is_production_env:
                self.logger.warning(f'[{response.status_code} - {e.request.url}]')
            else:
                self.logger.error(f'[{response.status_code} - {e.request.url}] {response.text}')
        except Exception as e:
            self.logger.trace(f'Error in response: {response.text}', e)
            raise e