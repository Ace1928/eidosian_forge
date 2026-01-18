from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
class RecognitionAlternative(param.Parameterized):
    """The RecognitionAlternative represents a word or
    sentence that has been recognised by the speech recognition service.

    Wraps the HTML5 SpeechRecognitionAlternative API

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognitionAlternative
    """
    confidence = param.Number(bounds=(0.0, 1.0), constant=True, doc='\n        A numeric estimate between 0 and 1 of how confident the speech recognition\n        system is that the recognition is correct.')
    transcript = param.String(constant=True, doc='\n        The transcript of the recognised word or sentence.')