import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
class BearlyInterpreterToolArguments(BaseModel):
    """Arguments for the BearlyInterpreterTool."""
    python_code: str = Field(..., example="print('Hello World')", description='The pure python script to be evaluated. The contents will be in main.py. It should not be in markdown format.')