from dataclasses import dataclass
from typing import Any, Optional
from .base import BaseInferenceType
@dataclass
class VisualQuestionAnsweringParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Visual Question Answering
    """
    top_k: Optional[int] = None
    'The number of answers to return (will be chosen by order of likelihood). Note that we\n    return less than topk answers if there are not enough options available within the\n    context.\n    '