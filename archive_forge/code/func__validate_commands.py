import logging
import platform
import warnings
from typing import Any, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
@root_validator
def _validate_commands(cls, values: dict) -> dict:
    """Validate commands."""
    commands = values.get('commands')
    if not isinstance(commands, list):
        values['commands'] = [commands]
    warnings.warn('The shell tool has no safeguards by default. Use at your own risk.')
    return values