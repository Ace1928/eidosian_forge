from __future__ import annotations
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import maybe_transform
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .....pagination import SyncCursorPage, AsyncCursorPage
from ....._base_client import (
from .....types.beta.threads.runs import RunStep, step_list_params
class Steps(SyncAPIResource):

    @cached_property
    def with_raw_response(self) -> StepsWithRawResponse:
        return StepsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> StepsWithStreamingResponse:
        return StepsWithStreamingResponse(self)

    def retrieve(self, step_id: str, *, thread_id: str, run_id: str, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> RunStep:
        """
        Retrieves a run step.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not thread_id:
            raise ValueError(f'Expected a non-empty value for `thread_id` but received {thread_id!r}')
        if not run_id:
            raise ValueError(f'Expected a non-empty value for `run_id` but received {run_id!r}')
        if not step_id:
            raise ValueError(f'Expected a non-empty value for `step_id` but received {step_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get(f'/threads/{thread_id}/runs/{run_id}/steps/{step_id}', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=RunStep)

    def list(self, run_id: str, *, thread_id: str, after: str | NotGiven=NOT_GIVEN, before: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, order: Literal['asc', 'desc'] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> SyncCursorPage[RunStep]:
        """
        Returns a list of run steps belonging to a run.

        Args:
          after: A cursor for use in pagination. `after` is an object ID that defines your place
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
        if not thread_id:
            raise ValueError(f'Expected a non-empty value for `thread_id` but received {thread_id!r}')
        if not run_id:
            raise ValueError(f'Expected a non-empty value for `run_id` but received {run_id!r}')
        extra_headers = {'OpenAI-Beta': 'assistants=v1', **(extra_headers or {})}
        return self._get_api_list(f'/threads/{thread_id}/runs/{run_id}/steps', page=SyncCursorPage[RunStep], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'before': before, 'limit': limit, 'order': order}, step_list_params.StepListParams)), model=RunStep)