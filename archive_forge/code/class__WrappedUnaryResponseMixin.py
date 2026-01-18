import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class _WrappedUnaryResponseMixin(Generic[P], _WrappedCall):

    def __await__(self) -> Iterator[P]:
        try:
            response = (yield from self._call.__await__())
            return response
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error