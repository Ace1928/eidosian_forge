from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
Allow arbitrary types.