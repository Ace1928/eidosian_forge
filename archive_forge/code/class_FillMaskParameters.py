from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class FillMaskParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Fill Mask
    """
    targets: Optional[List[str]] = None
    'When passed, the model will limit the scores to the passed targets instead of looking up\n    in the whole vocabulary. If the provided targets are not in the model vocab, they will be\n    tokenized and the first resulting token will be used (with a warning, and that might be\n    slower).\n    '
    top_k: Optional[int] = None
    'When passed, overrides the number of predictions to return.'