from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class AutomaticSpeechRecognitionOutput(BaseInferenceType):
    """Outputs of inference for the Automatic Speech Recognition task"""
    text: str
    'The recognized text.'
    chunks: Optional[List[AutomaticSpeechRecognitionOutputChunk]] = None
    'When returnTimestamps is enabled, chunks contains a list of audio chunks identified by\n    the model.\n    '