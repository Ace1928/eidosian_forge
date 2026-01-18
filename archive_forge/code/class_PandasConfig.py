import itertools
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import pyarrow as pa
import datasets
from datasets.table import table_cast
@dataclass
class PandasConfig(datasets.BuilderConfig):
    """BuilderConfig for Pandas."""
    features: Optional[datasets.Features] = None