from dataclasses import dataclass
from typing import Any, List, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TokenClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Token Classification
    """
    aggregation_strategy: Optional['TokenClassificationAggregationStrategy'] = None
    'The strategy used to fuse tokens based on model predictions'
    ignore_labels: Optional[List[str]] = None
    'A list of labels to ignore'
    stride: Optional[int] = None
    'The number of overlapping tokens between chunks when splitting the input text.'