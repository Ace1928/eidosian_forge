from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, cast
from argparse import ArgumentParser
from .._utils import get_client, print_model
from ..._types import NOT_GIVEN
from .._models import BaseModel
from .._progress import BufferReader
class CLITranscribeArgs(BaseModel):
    model: str
    file: str
    response_format: Optional[str] = None
    language: Optional[str] = None
    temperature: Optional[float] = None
    prompt: Optional[str] = None