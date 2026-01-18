import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class _KdtSuggestContext(BaseModel):
    """pydantic API request type"""
    table: Optional[str] = Field(default=None, title='Name of table')
    description: Optional[str] = Field(default=None, title='Table description')
    columns: List[str] = Field(default=None, title='Table columns list')
    rules: Optional[List[str]] = Field(default=None, title='Rules that apply to the table.')
    samples: Optional[Dict] = Field(default=None, title='Samples that apply to the entire context.')

    def to_system_str(self) -> str:
        lines = []
        lines.append(f'CREATE TABLE {self.table} AS')
        lines.append('(')
        if not self.columns or len(self.columns) == 0:
            ValueError("columns list can't be null.")
        columns = []
        for column in self.columns:
            column = column.replace('"', '').strip()
            columns.append(f'   {column}')
        lines.append(',\n'.join(columns))
        lines.append(');')
        if self.description:
            lines.append(f"COMMENT ON TABLE {self.table} IS '{self.description}';")
        if self.rules and len(self.rules) > 0:
            lines.append(f'-- When querying table {self.table} the following rules apply:')
            for rule in self.rules:
                lines.append(f'-- * {rule}')
        result = '\n'.join(lines)
        return result