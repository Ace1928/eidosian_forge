from __future__ import annotations
import logging
import os
import pathlib
import platform
from typing import Optional, Tuple
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.document_loaders.base import BaseLoader
class Framework(BaseModel):
    """Pebblo Framework instance.

    Args:
        name (str): Name of the Framework.
        version (str): Version of the Framework.
    """
    name: str
    version: str