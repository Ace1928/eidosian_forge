from __future__ import annotations
from typing import Union
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
from ...types.audio import speech_create_params
from ..._base_client import (
class SpeechWithRawResponse:

    def __init__(self, speech: Speech) -> None:
        self._speech = speech
        self.create = _legacy_response.to_raw_response_wrapper(speech.create)