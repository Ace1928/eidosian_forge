from asyncio import sleep as asleep
from time import sleep
from typing import Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class SleepInput(BaseModel):
    """Input for CopyFileTool."""
    sleep_time: int = Field(..., description='Time to sleep in seconds')