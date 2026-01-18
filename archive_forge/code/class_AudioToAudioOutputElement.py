from dataclasses import dataclass
from typing import Any
from .base import BaseInferenceType
@dataclass
class AudioToAudioOutputElement(BaseInferenceType):
    """Outputs of inference for the Audio To Audio task
    A generated audio file with its label.
    """
    blob: Any
    'The generated audio file.'
    content_type: str
    'The content type of audio file.'
    label: str
    'The label of the audio file.'