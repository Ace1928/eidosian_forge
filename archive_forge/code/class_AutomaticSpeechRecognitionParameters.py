from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class AutomaticSpeechRecognitionParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Automatic Speech Recognition
    """
    generate: Optional[AutomaticSpeechRecognitionGenerationParameters] = None
    'Parametrization of the text generation process'
    return_timestamps: Optional[bool] = None
    'Whether to output corresponding timestamps with the generated text'