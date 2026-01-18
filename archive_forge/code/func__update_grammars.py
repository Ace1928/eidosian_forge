from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
@param.depends('grammars', watch=True)
def _update_grammars(self):
    with param.edit_constant(self):
        if self.grammars:
            self._grammars = self.grammars.serialize()
        else:
            self._grammars = []