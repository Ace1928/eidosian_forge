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
class GradioModel(GradioBaseModel, BaseModel):

    @classmethod
    def from_json(cls, x) -> GradioModel:
        return cls(**x)