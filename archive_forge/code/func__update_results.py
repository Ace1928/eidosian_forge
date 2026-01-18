from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
@param.depends('results', watch=True)
def _update_results(self):
    with param.edit_constant(self):
        if self.results and 'alternatives' in self.results[-1]:
            self.value = self.results[-1]['alternatives'][0]['transcript'].lstrip()
        else:
            self.value = ''