from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
def add_from_string(self, src, weight=1.0):
    """
        Takes a src and weight and adds it to the GrammarList as a new
        Grammar object. The new Grammar object is returned.
        """
    grammar = Grammar(src=src, weight=weight)
    self.append(grammar)
    return grammar