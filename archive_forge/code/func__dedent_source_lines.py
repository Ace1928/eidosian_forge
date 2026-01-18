from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any
def _dedent_source_lines(source: list[str]) -> str:
    dedent_source = textwrap.dedent(''.join(source))
    if dedent_source.startswith((' ', '\t')):
        dedent_source = f'def dedent_workaround():\n{dedent_source}'
    return dedent_source