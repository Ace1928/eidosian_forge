from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
import yaml
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.utils import create_model
def _format_prompt_with_error_handling(self, inner_input: Dict) -> PromptValue:
    _inner_input = self._validate_input(inner_input)
    return self.format_prompt(**_inner_input)