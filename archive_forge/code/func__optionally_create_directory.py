from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def _optionally_create_directory(self, path: str) -> None:
    if self.ensure_exists:
        Path(path).mkdir(parents=True, exist_ok=True)