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
def _execute_sql(self, sql: str) -> Dict:
    """Execute an SQL query and return the result."""
    response = self.kdbc.execute_sql_and_decode(sql, limit=1, get_column_major=False)
    status_info = response['status_info']
    if status_info['status'] != 'OK':
        message = status_info['message']
        raise ValueError(message)
    records = response['records']
    if len(records) != 1:
        raise ValueError('No records returned.')
    record = records[0]
    response_dict = {}
    for col, val in record.items():
        response_dict[col] = val
    return response_dict