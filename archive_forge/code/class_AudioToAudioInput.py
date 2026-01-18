from dataclasses import dataclass
from typing import Any
from .base import BaseInferenceType
@dataclass
class AudioToAudioInput(BaseInferenceType):
    """Inputs for Audio to Audio inference"""
    inputs: Any
    'The input audio data'