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
class _StaticFiles:
    """
    Class to hold all static files for an app
    """
    all_paths = []

    def __init__(self, paths: list[str | pathlib.Path]) -> None:
        self.paths = paths
        self.all_paths = [pathlib.Path(p).resolve() for p in paths]

    @classmethod
    def clear(cls):
        cls.all_paths = []