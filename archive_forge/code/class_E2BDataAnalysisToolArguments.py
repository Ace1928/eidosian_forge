from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
class E2BDataAnalysisToolArguments(BaseModel):
    """Arguments for the E2BDataAnalysisTool."""
    python_code: str = Field(..., example="print('Hello World')", description='The python script to be evaluated. The contents will be in main.py. It should not be in markdown format.')