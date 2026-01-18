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
class ListFiles(GradioRootModel):
    root: List[FileData]

    def __getitem__(self, index):
        return self.root[index]

    def __iter__(self):
        return iter(self.root)