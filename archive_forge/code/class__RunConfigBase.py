import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@dataclass
class _RunConfigBase:
    """Parameters defining a run. A run is an evaluation of a triplet (model, dataset, metric) coupled with optimization parameters, allowing to compare a transformers baseline and a model optimized with Optimum."""
    metrics: List[str] = field(metadata={'description': 'List of metrics to evaluate on.'})