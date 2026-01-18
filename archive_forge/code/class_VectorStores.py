from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .files import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .file_batches import (
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import (
from ...._base_client import (
class VectorStores(SyncAPIResource):

    @cached_property
    def files(self) -> Files:
        return Files(self._client)

    @cached_property
    def file_batches(self) -> FileBatches:
        return FileBatches(self._client)

    @cached_property
    def with_raw_response(self) -> VectorStoresWithRawResponse:
        return VectorStoresWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> VectorStoresWithStreamingResponse:
        return VectorStoresWithStreamingResponse(self)

    def create(self, *, expires_after: vector_store_create_params.ExpiresAfter | NotGiven=NOT_GIVEN, file_ids: List[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, name: str | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> VectorStore:
        """
        Create a vector store.

        Args:
          expires_after: The expiration policy for a vector store.

          file_ids: A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that
              the vector store should use. Useful for tools like `file_search` that can access
              files.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          name: The name of the vector store.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v2', **(extra_headers or {})}
        return self._post('/vector_stores', body=maybe_transform({'expires_after': expires_after, 'file_ids': file_ids, 'metadata': metadata, 'name': name}, vector_store_create_params.VectorStoreCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=VectorStore)

    def retrieve(self, vector_store_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> VectorStore:
        """
        Retrieves a vector store.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not vector_store_id:
            raise ValueError(f'Expected a non-empty value for `vector_store_id` but received {vector_store_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v2', **(extra_headers or {})}
        return self._get(f'/vector_stores/{vector_store_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=VectorStore)

    def update(self, vector_store_id: str, *, expires_after: Optional[vector_store_update_params.ExpiresAfter] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, name: Optional[str] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> VectorStore:
        """
        Modifies a vector store.

        Args:
          expires_after: The expiration policy for a vector store.

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format. Keys
              can be a maximum of 64 characters long and values can be a maxium of 512
              characters long.

          name: The name of the vector store.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not vector_store_id:
            raise ValueError(f'Expected a non-empty value for `vector_store_id` but received {vector_store_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v2', **(extra_headers or {})}
        return self._post(f'/vector_stores/{vector_store_id}', body=maybe_transform({'expires_after': expires_after, 'metadata': metadata, 'name': name}, vector_store_update_params.VectorStoreUpdateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=VectorStore)

    def list(self, *, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[VectorStore]:
        """Returns a list of vector stores.

        Args:
          after: A cursor for use in pagination.

        `after` is an object ID that defines your place
              in the list. For instance, if you make a list request and receive 100 objects,
              ending with obj_foo, your subsequent call can include after=obj_foo in order to
              fetch the next page of the list.

          before: A cursor for use in pagination. `before` is an object ID that defines your place
              in the list. For instance, if you make a list request and receive 100 objects,
              ending with obj_foo, your subsequent call can include before=obj_foo in order to
              fetch the previous page of the list.

          limit: A limit on the number of objects to be returned. Limit can range between 1 and
              100, and the default is 20.

          order: Sort order by the `created_at` timestamp of the objects. `asc` for ascending
              order and `desc` for descending order.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'OpenAI-Beta': 'assistants=v2', **(extra_headers or {})}
        return self._get_api_list('/vector_stores', page=SyncCursorPage[VectorStore], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, vector_store_list_params.VectorStoreListParams)), model=VectorStore)

    def delete(self, vector_store_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> VectorStoreDeleted:
        """
        Delete a vector store.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not vector_store_id:
            raise ValueError(f'Expected a non-empty value for `vector_store_id` but received {vector_store_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v2', **(extra_headers or {})}
        return self._delete(f'/vector_stores/{vector_store_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=VectorStoreDeleted)