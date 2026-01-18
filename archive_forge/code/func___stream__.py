from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
def __stream__(self) -> Iterator[_T]:
    cast_to = cast(Any, self._cast_to)
    response = self.response
    process_data = self._client._process_response_data
    iterator = self._iter_events()
    for sse in iterator:
        if sse.data.startswith('[DONE]'):
            break
        if sse.event is None:
            data = sse.json()
            if is_mapping(data) and data.get('error'):
                message = None
                error = data.get('error')
                if is_mapping(error):
                    message = error.get('message')
                if not message or not isinstance(message, str):
                    message = 'An error occurred during streaming'
                raise APIError(message=message, request=self.response.request, body=data['error'])
            yield process_data(data=data, cast_to=cast_to, response=response)
        else:
            data = sse.json()
            if sse.event == 'error' and is_mapping(data) and data.get('error'):
                message = None
                error = data.get('error')
                if is_mapping(error):
                    message = error.get('message')
                if not message or not isinstance(message, str):
                    message = 'An error occurred during streaming'
                raise APIError(message=message, request=self.response.request, body=data['error'])
            yield process_data(data={'data': data, 'event': sse.event}, cast_to=cast_to, response=response)
    for _sse in iterator:
        ...