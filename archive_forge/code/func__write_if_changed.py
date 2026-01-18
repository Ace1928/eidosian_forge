import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
def _write_if_changed(self, filename: str, contents: str) -> None:
    old_contents: Optional[str]
    try:
        with open(filename) as f:
            old_contents = f.read()
    except OSError:
        old_contents = None
    if contents != old_contents:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(contents)