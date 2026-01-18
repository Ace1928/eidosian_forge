from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
class AnsiDecorator:
    plain_code = '\x1b[0m'

    def __init__(self, text: str, code: str, quoted: bool=False):
        self.text = text
        self.code = code
        self.quoted = quoted

    def get_text(self, with_codes: bool) -> str:
        text = self.text
        if with_codes and self.code:
            text = self.code + self.text + AnsiDecorator.plain_code
        if self.quoted:
            text = f'"{text}"'
        return text

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return self.get_text(colorize_console())