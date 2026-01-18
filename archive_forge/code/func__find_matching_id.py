import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union
import requests
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
def _find_matching_id(self, uuid: str) -> Union[str, None]:
    for id, result in self._results.items():
        if result['uuid'] == uuid:
            return id
    return None