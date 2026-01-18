from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
def add_from_uri(self, uri, weight=1.0):
    """
        Takes a grammar present at a specific uri, and adds it to the
        GrammarList as a new Grammar object. The new Grammar object is
        returned.
        """
    grammar = Grammar(uri=uri, weight=weight)
    self.append(grammar)
    return grammar