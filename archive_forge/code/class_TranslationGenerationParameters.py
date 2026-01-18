from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from .base import BaseInferenceType
@dataclass
class TranslationGenerationParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text2text Generation
    """
    clean_up_tokenization_spaces: Optional[bool] = None
    'Whether to clean up the potential extra spaces in the text output.'
    generate_parameters: Optional[Dict[str, Any]] = None
    'Additional parametrization of the text generation algorithm'
    truncation: Optional['TranslationGenerationTruncationStrategy'] = None
    'The truncation strategy to use'