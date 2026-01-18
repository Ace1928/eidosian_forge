from __future__ import annotations
import hashlib
import os
import typing
import urllib.parse
import warnings
from dataclasses import dataclass, field
import fastapi
from fastapi.responses import RedirectResponse
from huggingface_hub import HfFolder, whoami
from .utils import get_space
def _redirect_to_target(request: fastapi.Request, default_target: str='/') -> RedirectResponse:
    target = request.query_params.get('_target_url', default_target)
    return RedirectResponse(target)