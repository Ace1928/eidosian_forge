import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """Get the hours passed between two datetimes."""
    return (time - ref_time).total_seconds() / 3600