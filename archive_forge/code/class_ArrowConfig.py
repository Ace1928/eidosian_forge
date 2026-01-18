import itertools
from dataclasses import dataclass
from typing import Optional
import pyarrow as pa
import datasets
from datasets.table import table_cast
@dataclass
class ArrowConfig(datasets.BuilderConfig):
    """BuilderConfig for Arrow."""
    features: Optional[datasets.Features] = None