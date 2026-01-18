from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
class RecognitionResult(param.Parameterized):
    """The Result represents a single recognition match, which may contain
    multiple RecognitionAlternative objects.

    Wraps the HTML5 SpeechRecognitionResult API.

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognitionResult
    """
    alternatives = param.List(item_type=RecognitionAlternative, constant=True, doc='\n        The list of the n-best alternatives')
    is_final = param.Boolean(constant=True, doc='\n        A Boolean that states whether this result is final (True) or\n        not (False) — if so, then this is the final time this result\n        will be returned; if not, then this result is an interim\n        result, and may be updated later on.')

    @classmethod
    def create_from_dict(cls, result):
        """
        Deserializes a serialized RecognitionResult
        """
        result = result.copy()
        alternatives = result.get('alternatives', [])
        _alternatives = []
        for alternative in alternatives:
            _alternatives.append(RecognitionAlternative(**alternative))
        result['alternatives'] = _alternatives
        return cls(**result)

    @classmethod
    def create_from_list(cls, results):
        """
        Deserializes a list of serialized RecognitionResults.
        """
        return [cls.create_from_dict(result) for result in results]