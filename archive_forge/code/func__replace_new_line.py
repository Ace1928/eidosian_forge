from __future__ import annotations
import json
import re
from typing import Any, Callable, List
from langchain_core.exceptions import OutputParserException
def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub('\\n', '\\\\n', value)
    value = re.sub('\\r', '\\\\r', value)
    value = re.sub('\\t', '\\\\t', value)
    value = re.sub('(?<!\\\\)"', '\\"', value)
    return match.group(1) + value + match.group(3)