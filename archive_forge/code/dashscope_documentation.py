from __future__ import annotations
import logging
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
Call out to DashScope's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        