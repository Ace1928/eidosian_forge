import itertools
import warnings
from dataclasses import InitVar, dataclass
from io import StringIO
from typing import Optional
import pyarrow as pa
import datasets
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
@dataclass
class TextConfig(datasets.BuilderConfig):
    """BuilderConfig for text files."""
    features: Optional[datasets.Features] = None
    encoding: str = 'utf-8'
    errors: InitVar[Optional[str]] = 'deprecated'
    encoding_errors: Optional[str] = None
    chunksize: int = 10 << 20
    keep_linebreaks: bool = False
    sample_by: str = 'line'

    def __post_init__(self, errors):
        if errors != 'deprecated':
            warnings.warn(f"'errors' was deprecated in favor of 'encoding_errors' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'encoding_errors={errors}' instead.", FutureWarning)
            self.encoding_errors = errors