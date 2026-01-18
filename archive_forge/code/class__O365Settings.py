from __future__ import annotations
import logging
import os
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Union
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
from langchain_community.document_loaders.blob_loaders.schema import Blob
class _O365Settings(BaseSettings):
    client_id: str = Field(..., env='O365_CLIENT_ID')
    client_secret: SecretStr = Field(..., env='O365_CLIENT_SECRET')

    class Config:
        env_prefix = ''
        case_sentive = False
        env_file = '.env'