from __future__ import annotations
from typing import Dict, Optional
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import Batch, batch_list_params, batch_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ..pagination import SyncCursorPage, AsyncCursorPage
from .._base_client import (
class AsyncBatches(AsyncAPIResource):

    @cached_property
    def with_raw_response(self) -> AsyncBatchesWithRawResponse:
        return AsyncBatchesWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncBatchesWithStreamingResponse:
        return AsyncBatchesWithStreamingResponse(self)

    async def create(self, *, completion_window: Literal['24h'], endpoint: Literal['/v1/chat/completions'], input_file_id: str, metadata: Optional[Dict[str, str]] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Batch:
        """
        Creates and executes a batch from an uploaded file of requests

        Args:
          completion_window: The time frame within which the batch should be processed. Currently only `24h`
              is supported.

          endpoint: The endpoint to be used for all requests in the batch. Currently only
              `/v1/chat/completions` is supported.

          input_file_id: The ID of an uploaded file that contains requests for the new batch.

              See [upload file](https://platform.openai.com/docs/api-reference/files/create)
              for how to upload a file.

              Your input file must be formatted as a JSONL file, and must be uploaded with the
              purpose `batch`.

          metadata: Optional custom metadata for the batch.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post('/batches', body=await async_maybe_transform({'completion_window': completion_window, 'endpoint': endpoint, 'input_file_id': input_file_id, 'metadata': metadata}, batch_create_params.BatchCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Batch)

    async def retrieve(self, batch_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Batch:
        """
        Retrieves a batch.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not batch_id:
            raise ValueError(f'Expected a non-empty value for `batch_id` but received {batch_id!r}')
        return await self._get(f'/batches/{batch_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Batch)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[Batch, AsyncCursorPage[Batch]]:
        """List your organization's batches.

        Args:
          after: A cursor for use in pagination.

        `after` is an object ID that defines your place
              in the list. For instance, if you make a list request and receive 100 objects,
              ending with obj_foo, your subsequent call can include after=obj_foo in order to
              fetch the next page of the list.

          limit: A limit on the number of objects to be returned. Limit can range between 1 and
              100, and the default is 20.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get_api_list('/batches', page=AsyncCursorPage[Batch], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, batch_list_params.BatchListParams)), model=Batch)

    async def cancel(self, batch_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Batch:
        """
        Cancels an in-progress batch.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not batch_id:
            raise ValueError(f'Expected a non-empty value for `batch_id` but received {batch_id!r}')
        return await self._post(f'/batches/{batch_id}/cancel', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Batch)