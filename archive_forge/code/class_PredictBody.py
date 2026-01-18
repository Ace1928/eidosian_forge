from __future__ import annotations
import pathlib
import secrets
import shutil
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from fastapi import Request
from gradio_client.utils import traverse
from . import wasm_utils
class PredictBody(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    session_hash: Optional[str] = None
    event_id: Optional[str] = None
    data: List[Any]
    event_data: Optional[Any] = None
    fn_index: Optional[int] = None
    trigger_id: Optional[int] = None
    simple_format: bool = False
    batched: Optional[bool] = False
    request: Optional[Request] = None