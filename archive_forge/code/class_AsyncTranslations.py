from __future__ import annotations
from typing import Union, Mapping, cast
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...types.audio import Translation, translation_create_params
from ..._base_client import (
class AsyncTranslations(AsyncAPIResource):

    @cached_property
    def with_raw_response(self) -> AsyncTranslationsWithRawResponse:
        return AsyncTranslationsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTranslationsWithStreamingResponse:
        return AsyncTranslationsWithStreamingResponse(self)

    async def create(self, *, file: FileTypes, model: Union[str, Literal['whisper-1']], prompt: str | NotGiven=NOT_GIVEN, response_format: str | NotGiven=NOT_GIVEN, temperature: float | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Translation:
        """
        Translates audio into English.

        Args:
          file: The audio file object (not file name) translate, in one of these formats: flac,
              mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

          model: ID of the model to use. Only `whisper-1` (which is powered by our open source
              Whisper V2 model) is currently available.

          prompt: An optional text to guide the model's style or continue a previous audio
              segment. The
              [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)
              should be in English.

          response_format: The format of the transcript output, in one of these options: `json`, `text`,
              `srt`, `verbose_json`, or `vtt`.

          temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the
              output more random, while lower values like 0.2 will make it more focused and
              deterministic. If set to 0, the model will use
              [log probability](https://en.wikipedia.org/wiki/Log_probability) to
              automatically increase the temperature until certain thresholds are hit.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal({'file': file, 'model': model, 'prompt': prompt, 'response_format': response_format, 'temperature': temperature})
        files = extract_files(cast(Mapping[str, object], body), paths=[['file']])
        if files:
            extra_headers = {'Content-Type': 'multipart/form-data', **(extra_headers or {})}
        return await self._post('/audio/translations', body=await async_maybe_transform(body, translation_create_params.TranslationCreateParams), files=files, options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Translation)