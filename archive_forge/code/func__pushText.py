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
def _pushText(self, id: str, text: str) -> str:
    field = {'textfield': {'text': {'body': text, 'format': 0}}, 'processing_options': {'ml_text': self._config['enable_ml']}}
    return self._pushField(id, field)