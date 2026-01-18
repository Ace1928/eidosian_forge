import builtins
import json
from enum import Enum
from typing import List, Optional, Type, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.ainetwork.base import AINBaseTool
class AppOperationType(str, Enum):
    """Type of app operation as enumerator."""
    SET_ADMIN = 'SET_ADMIN'
    GET_CONFIG = 'GET_CONFIG'