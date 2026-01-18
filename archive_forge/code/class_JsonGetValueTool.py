from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
from langchain_core.tools import BaseTool
class JsonGetValueTool(BaseTool):
    """Tool for getting a value in a JSON spec."""
    name: str = 'json_spec_get_value'
    description: str = '\n    Can be used to see value in string format at a given path.\n    Before calling this you should be SURE that the path to this exists.\n    The input is a text representation of the path to the dict in Python syntax (e.g. data["key1"][0]["key2"]).\n    '
    spec: JsonSpec

    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        return self.spec.value(tool_input)

    async def _arun(self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        return self._run(tool_input)