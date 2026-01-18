import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@generate_doc_dataclass
@dataclass
class BenchmarkTimeArgs:
    """Parameters related to time benchmark."""
    duration: Optional[int] = field(default=30, metadata={'description': 'Duration in seconds of the time evaluation.'})
    warmup_runs: Optional[int] = field(default=10, metadata={'description': 'Number of warmup calls to forward() before the time evaluation.'})