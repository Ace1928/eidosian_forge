from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
from argparse import ArgumentParser
from .._utils import get_client, print_model
from ..._types import NOT_GIVEN, NotGiven, NotGivenOr
from .._models import BaseModel
from .._progress import BufferReader
class CLIImageCreateArgs(BaseModel):
    prompt: str
    num_images: int
    size: str
    response_format: str
    model: NotGivenOr[str] = NOT_GIVEN