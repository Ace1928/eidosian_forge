import base64
import io
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from ..constants import ENDPOINT
from ..utils import (
from ._text_generation import TextGenerationStreamResponse, _parse_text_generation_error
def _fetch_recommended_models() -> Dict[str, Optional[str]]:
    global _RECOMMENDED_MODELS
    if _RECOMMENDED_MODELS is None:
        response = get_session().get(f'{ENDPOINT}/api/tasks', headers=build_hf_headers())
        hf_raise_for_status(response)
        _RECOMMENDED_MODELS = {task: _first_or_none(details['widgetModels']) for task, details in response.json().items()}
    return _RECOMMENDED_MODELS