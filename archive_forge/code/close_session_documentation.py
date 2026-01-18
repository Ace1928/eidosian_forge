from typing import TYPE_CHECKING, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
Tool that closes an existing Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "close_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments. Default: UpdateSessionSchema
    