from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
class FromFile(Implementation):
    """ A custom model implementation read from a separate source file.

    Args:
        path (str) :
            The path to the file containing the extension source code

    """

    def __init__(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            self.code = f.read()
        self.file = path

    @property
    def lang(self) -> str:
        if self.file is not None:
            if self.file.endswith('.ts'):
                return 'typescript'
            if self.file.endswith('.js'):
                return 'javascript'
            if self.file.endswith(('.css', '.less')):
                return 'less'
        raise ValueError(f'unknown file type {self.file}')