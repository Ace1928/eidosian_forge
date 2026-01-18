from dataclasses import dataclass
from typing import Any, List, Optional
from .base import BaseInferenceType
@dataclass
class ZeroShotImageClassificationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Zero Shot Image Classification
    """
    hypothesis_template: Optional[str] = None
    'The sentence used in conjunction with candidateLabels to attempt the text classification\n    by replacing the placeholder with the candidate labels.\n    '