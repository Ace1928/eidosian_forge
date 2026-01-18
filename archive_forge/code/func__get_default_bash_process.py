import logging
import platform
import warnings
from typing import Any, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
def _get_default_bash_process() -> Any:
    """Get default bash process."""
    try:
        from langchain_experimental.llm_bash.bash import BashProcess
    except ImportError:
        raise ImportError('BashProcess has been moved to langchain experimental.To use this tool, install langchain-experimental with `pip install langchain-experimental`.')
    return BashProcess(return_err_output=True)