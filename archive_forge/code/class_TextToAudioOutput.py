from dataclasses import dataclass
from typing import Any, Literal, Optional, Union
from .base import BaseInferenceType
@dataclass
class TextToAudioOutput(BaseInferenceType):
    """Outputs of inference for the Text To Audio task"""
    audio: Any
    'The generated audio waveform.'
    sampling_rate: Any
    text_to_audio_output_sampling_rate: Optional[float] = None
    'The sampling rate of the generated audio waveform.'