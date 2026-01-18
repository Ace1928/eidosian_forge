import copy
import json
from typing import Any, Dict, List, Optional, Type, Union
import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs.chat_generation import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
The name of the attribute to return.