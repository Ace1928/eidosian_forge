import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utilities.vertexai import get_client_info
@dataclass
class DocAIParsingResults:
    """A dataclass to store Document AI parsing results."""
    source_path: str
    parsed_path: str