from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class AutomaticSpeechRecognitionOutputChunk(BaseInferenceType):
    text: str
    'A chunk of text identified by the model'
    timestamps: List[float]
    'The start and end timestamps corresponding with the text'