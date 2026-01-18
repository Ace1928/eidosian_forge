from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class TextToAudioParameters(BaseInferenceType):
    """Additional inference parameters
    Additional inference parameters for Text To Audio
    """
    generate: Optional[TextToAudioGenerationParameters] = None
    'Parametrization of the text generation process'