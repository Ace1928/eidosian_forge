import logging
import platform
import warnings
from typing import Any, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
def _get_platform() -> str:
    """Get platform."""
    system = platform.system()
    if system == 'Darwin':
        return 'MacOS'
    return system