from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
@param.depends('lang', watch=True)
def _handle_lang_changed(self):
    if not self._voices_by_language or not self.lang:
        self.param.voice.default = None
        self.voice = None
        self.param.voice.objects = []
        return
    voices = self._voices_by_language[self.lang]
    if self.voice and self.voice in voices:
        default_voice = self.voice
    else:
        default_voice = voices[0]
        for voice in voices:
            if voice.default:
                default_voice = voice
    self.param.voice.objects = voices
    self.param.voice.default = default_voice
    self.voice = default_voice