from __future__ import annotations
import logging # isort:skip
import itertools
import re
from dataclasses import dataclass
from typing import (
from ..models import (
from ..models.tools import (
from ..util.warnings import warn
def _resolve_tools(tools: str | Sequence[Tool | str]) -> tuple[list[Tool], dict[str, Tool]]:
    tool_objs: list[Tool] = []
    tool_map: dict[str, Tool] = {}
    if not isinstance(tools, str):
        temp_tool_str = ''
        for tool in tools:
            if isinstance(tool, Tool):
                tool_objs.append(tool)
            elif isinstance(tool, str):
                temp_tool_str += tool + ','
            else:
                raise ValueError('tool should be a string or an instance of Tool class')
        tools = temp_tool_str
    for tool in re.split('\\s*,\\s*', tools.strip()):
        if tool == '':
            continue
        tool_obj = Tool.from_string(tool)
        tool_objs.append(tool_obj)
        tool_map[tool] = tool_obj
    return (tool_objs, tool_map)