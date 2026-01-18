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
class BaseClient(Generic[_HttpxClientT, _DefaultStreamT]):
    _client: _HttpxClientT
    _version: str
    _base_url: URL
    max_retries: int
    timeout: Union[float, Timeout, None]
    _limits: httpx.Limits
    _proxies: ProxiesTypes | None
    _transport: Transport | AsyncTransport | None
    _strict_response_validation: bool
    _idempotency_header: str | None
    _default_stream_cls: type[_DefaultStreamT] | None = None

    def __init__(self, *, version: str, base_url: str | URL, _strict_response_validation: bool, max_retries: int=DEFAULT_MAX_RETRIES, timeout: float | Timeout | None=DEFAULT_TIMEOUT, limits: httpx.Limits, transport: Transport | AsyncTransport | None, proxies: ProxiesTypes | None, custom_headers: Mapping[str, str] | None=None, custom_query: Mapping[str, object] | None=None) -> None:
        self._version = version
        self._base_url = self._enforce_trailing_slash(URL(base_url))
        self.max_retries = max_retries
        self.timeout = timeout
        self._limits = limits
        self._proxies = proxies
        self._transport = transport
        self._custom_headers = custom_headers or {}
        self._custom_query = custom_query or {}
        self._strict_response_validation = _strict_response_validation
        self._idempotency_header = None

    def _enforce_trailing_slash(self, url: URL) -> URL:
        if url.raw_path.endswith(b'/'):
            return url
        return url.copy_with(raw_path=url.raw_path + b'/')

    def _make_status_error_from_response(self, response: httpx.Response) -> APIStatusError:
        if response.is_closed and (not response.is_stream_consumed):
            body = None
            err_msg = f'Error code: {response.status_code}'
        else:
            err_text = response.text.strip()
            body = err_text
            try:
                body = json.loads(err_text)
                err_msg = f'Error code: {response.status_code} - {body}'
            except Exception:
                err_msg = err_text or f'Error code: {response.status_code}'
        return self._make_status_error(err_msg, body=body, response=response)

    def _make_status_error(self, err_msg: str, *, body: object, response: httpx.Response) -> _exceptions.APIStatusError:
        raise NotImplementedError()

    def _remaining_retries(self, remaining_retries: Optional[int], options: FinalRequestOptions) -> int:
        return remaining_retries if remaining_retries is not None else options.get_max_retries(self.max_retries)

    def _build_headers(self, options: FinalRequestOptions) -> httpx.Headers:
        custom_headers = options.headers or {}
        headers_dict = _merge_mappings(self.default_headers, custom_headers)
        self._validate_headers(headers_dict, custom_headers)
        headers = httpx.Headers(headers_dict)
        idempotency_header = self._idempotency_header
        if idempotency_header and options.method.lower() != 'get' and (idempotency_header not in headers):
            headers[idempotency_header] = options.idempotency_key or self._idempotency_key()
        return headers

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

    def _make_sse_decoder(self) -> SSEDecoder | SSEBytesDecoder:
        return SSEDecoder()

    def _build_request(self, options: FinalRequestOptions) -> httpx.Request:
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Request options: %s', model_dump(options, exclude_unset=True))
        kwargs: dict[str, Any] = {}
        json_data = options.json_data
        if options.extra_json is not None:
            if json_data is None:
                json_data = cast(Body, options.extra_json)
            elif is_mapping(json_data):
                json_data = _merge_mappings(json_data, options.extra_json)
            else:
                raise RuntimeError(f'Unexpected JSON data type, {type(json_data)}, cannot merge with `extra_body`')
        headers = self._build_headers(options)
        params = _merge_mappings(self._custom_query, options.params)
        content_type = headers.get('Content-Type')
        if content_type is not None and content_type.startswith('multipart/form-data'):
            if 'boundary' not in content_type:
                headers.pop('Content-Type')
            if json_data:
                if not is_dict(json_data):
                    raise TypeError(f'Expected query input to be a dictionary for multipart requests but got {type(json_data)} instead.')
                kwargs['data'] = self._serialize_multipartform(json_data)
        return self._client.build_request(headers=headers, timeout=self.timeout if isinstance(options.timeout, NotGiven) else options.timeout, method=options.method, url=self._prepare_url(options.url), params=self.qs.stringify(cast(Mapping[str, Any], params)) if params else None, json=json_data, files=options.files, **kwargs)

    def _serialize_multipartform(self, data: Mapping[object, object]) -> dict[str, object]:
        items = self.qs.stringify_items(data, array_format='brackets')
        serialized: dict[str, object] = {}
        for key, value in items:
            existing = serialized.get(key)
            if not existing:
                serialized[key] = value
                continue
            if is_list(existing):
                existing.append(value)
            else:
                serialized[key] = [existing, value]
        return serialized

    def _maybe_override_cast_to(self, cast_to: type[ResponseT], options: FinalRequestOptions) -> type[ResponseT]:
        if not is_given(options.headers):
            return cast_to
        headers = dict(options.headers)
        override_cast_to = headers.pop(OVERRIDE_CAST_TO_HEADER, NOT_GIVEN)
        if is_given(override_cast_to):
            options.headers = headers
            return cast(Type[ResponseT], override_cast_to)
        return cast_to

    def _should_stream_response_body(self, request: httpx.Request) -> bool:
        return request.headers.get(RAW_RESPONSE_HEADER) == 'stream'

    def _process_response_data(self, *, data: object, cast_to: type[ResponseT], response: httpx.Response) -> ResponseT:
        if data is None:
            return cast(ResponseT, None)
        if cast_to is object:
            return cast(ResponseT, data)
        try:
            if inspect.isclass(cast_to) and issubclass(cast_to, ModelBuilderProtocol):
                return cast(ResponseT, cast_to.build(response=response, data=data))
            if self._strict_response_validation:
                return cast(ResponseT, validate_type(type_=cast_to, value=data))
            return cast(ResponseT, construct_type(type_=cast_to, value=data))
        except pydantic.ValidationError as err:
            raise APIResponseValidationError(response=response, body=data) from err

    @property
    def qs(self) -> Querystring:
        return Querystring()

    @property
    def custom_auth(self) -> httpx.Auth | None:
        return None

    @property
    def auth_headers(self) -> dict[str, str]:
        return {}

    @property
    def default_headers(self) -> dict[str, str | Omit]:
        return {'Accept': 'application/json', 'Content-Type': 'application/json', 'User-Agent': self.user_agent, **self.platform_headers(), **self.auth_headers, **self._custom_headers}

    def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
        """Validate the given default headers and custom headers.

        Does nothing by default.
        """
        return

    @property
    def user_agent(self) -> str:
        return f'{self.__class__.__name__}/Python {self._version}'

    @property
    def base_url(self) -> URL:
        return self._base_url

    @base_url.setter
    def base_url(self, url: URL | str) -> None:
        self._base_url = self._enforce_trailing_slash(url if isinstance(url, URL) else URL(url))

    def platform_headers(self) -> Dict[str, str]:
        return platform_headers(self._version)

    def _parse_retry_after_header(self, response_headers: Optional[httpx.Headers]=None) -> float | None:
        """Returns a float of the number of seconds (not milliseconds) to wait after retrying, or None if unspecified.

        About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        See also  https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax
        """
        if response_headers is None:
            return None
        try:
            retry_ms_header = response_headers.get('retry-after-ms', None)
            return float(retry_ms_header) / 1000
        except (TypeError, ValueError):
            pass
        retry_header = response_headers.get('retry-after')
        try:
            return float(retry_header)
        except (TypeError, ValueError):
            pass
        retry_date_tuple = email.utils.parsedate_tz(retry_header)
        if retry_date_tuple is None:
            return None
        retry_date = email.utils.mktime_tz(retry_date_tuple)
        return float(retry_date - time.time())

    def _calculate_retry_timeout(self, remaining_retries: int, options: FinalRequestOptions, response_headers: Optional[httpx.Headers]=None) -> float:
        max_retries = options.get_max_retries(self.max_retries)
        retry_after = self._parse_retry_after_header(response_headers)
        if retry_after is not None and 0 < retry_after <= 60:
            return retry_after
        nb_retries = max_retries - remaining_retries
        sleep_seconds = min(INITIAL_RETRY_DELAY * pow(2.0, nb_retries), MAX_RETRY_DELAY)
        jitter = 1 - 0.25 * random()
        timeout = sleep_seconds * jitter
        return timeout if timeout >= 0 else 0

    def _should_retry(self, response: httpx.Response) -> bool:
        should_retry_header = response.headers.get('x-should-retry')
        if should_retry_header == 'true':
            log.debug('Retrying as header `x-should-retry` is set to `true`')
            return True
        if should_retry_header == 'false':
            log.debug('Not retrying as header `x-should-retry` is set to `false`')
            return False
        if response.status_code == 408:
            log.debug('Retrying due to status code %i', response.status_code)
            return True
        if response.status_code == 409:
            log.debug('Retrying due to status code %i', response.status_code)
            return True
        if response.status_code == 429:
            log.debug('Retrying due to status code %i', response.status_code)
            return True
        if response.status_code >= 500:
            log.debug('Retrying due to status code %i', response.status_code)
            return True
        log.debug('Not retrying')
        return False

    def _idempotency_key(self) -> str:
        return f'stainless-python-retry-{uuid.uuid4()}'