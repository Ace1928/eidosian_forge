import os
from enum import Enum
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
class SearchDepth(Enum):
    """Search depth as enumerator."""
    BASIC = 'basic'
    ADVANCED = 'advanced'