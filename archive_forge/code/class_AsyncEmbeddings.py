from __future__ import annotations
import base64
from typing import List, Union, Iterable, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import CreateEmbeddingResponse, embedding_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import is_given, maybe_transform
from .._compat import cached_property
from .._extras import numpy as np, has_numpy
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (
class AsyncEmbeddings(AsyncAPIResource):

    @cached_property
    def with_raw_response(self) -> AsyncEmbeddingsWithRawResponse:
        return AsyncEmbeddingsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncEmbeddingsWithStreamingResponse:
        return AsyncEmbeddingsWithStreamingResponse(self)

    async def create(self, *, input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]], model: Union[str, Literal['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']], dimensions: int | NotGiven=NOT_GIVEN, encoding_format: Literal['float', 'base64'] | NotGiven=NOT_GIVEN, user: str | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> CreateEmbeddingResponse:
        """
        Creates an embedding vector representing the input text.

        Args:
          input: Input text to embed, encoded as a string or array of tokens. To embed multiple
              inputs in a single request, pass an array of strings or array of token arrays.
              The input must not exceed the max input tokens for the model (8192 tokens for
              `text-embedding-ada-002`), cannot be an empty string, and any array must be 2048
              dimensions or less.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          model: ID of the model to use. You can use the
              [List models](https://platform.openai.com/docs/api-reference/models/list) API to
              see all of your available models, or see our
              [Model overview](https://platform.openai.com/docs/models/overview) for
              descriptions of them.

          dimensions: The number of dimensions the resulting output embeddings should have. Only
              supported in `text-embedding-3` and later models.

          encoding_format: The format to return the embeddings in. Can be either `float` or
              [`base64`](https://pypi.org/project/pybase64/).

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        params = {'input': input, 'model': model, 'user': user, 'dimensions': dimensions, 'encoding_format': encoding_format}
        if not is_given(encoding_format) and has_numpy():
            params['encoding_format'] = 'base64'

        def parser(obj: CreateEmbeddingResponse) -> CreateEmbeddingResponse:
            if is_given(encoding_format):
                return obj
            for embedding in obj.data:
                data = cast(object, embedding.embedding)
                if not isinstance(data, str):
                    continue
                embedding.embedding = np.frombuffer(base64.b64decode(data), dtype='float32').tolist()
            return obj
        return await self._post('/embeddings', body=maybe_transform(params, embedding_create_params.EmbeddingCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, post_parser=parser), cast_to=CreateEmbeddingResponse)