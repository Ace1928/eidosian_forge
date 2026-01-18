from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
class Utterance(param.Parameterized):
    """
    An *utterance* is the smallest unit of speech in spoken language analysis.

    The Utterance Model wraps the HTML5 SpeechSynthesisUtterance API

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisUtterance
    """
    value = param.String(default='', doc='\n        The text that will be synthesised when the utterance is\n        spoken. The text may be provided as plain text, or a\n        well-formed SSML document.')
    lang = param.ObjectSelector(default='', doc='\n        The language of the utterance.')
    pitch = param.Number(default=1.0, bounds=(0.0, 2.0), doc='\n        The pitch at which the utterance will be spoken at expressed\n        as a number between 0 and 2.')
    rate = param.Number(default=1.0, bounds=(0.1, 10.0), doc='\n        The speed at which the utterance will be spoken at expressed\n        as a number between 0.1 and 10.')
    voice = param.ObjectSelector(doc='\n        The voice that will be used to speak the utterance.')
    volume = param.Number(default=1.0, bounds=(0.0, 1.0), doc=' The\n        volume that the utterance will be spoken at expressed as a\n        number between 0 and 1.')

    def __init__(self, **params):
        voices = params.pop('voices', [])
        super().__init__(**params)
        self._voices_by_language = {}
        self.set_voices(voices)

    def to_dict(self, include_uuid=True):
        """Returns the object parameter values in a dictionary

        Returns:
            Dict: [description]
        """
        result = {'lang': self.lang, 'pitch': self.pitch, 'rate': self.rate, 'text': self.value, 'volume': self.volume}
        if self.voice and self.voice.name:
            result['voice'] = self.voice.name
        if include_uuid:
            result['uuid'] = str(uuid.uuid4())
        return result

    def set_voices(self, voices):
        """Updates the `lang` and `voice` parameter objects, default and value"""
        if not voices:
            self.param.lang.objects = ['en-US']
            self.param.lang.default = 'en-US'
            self.lang = 'en-US'
            return
        self._voices_by_language = Voice.group_by_lang(voices)
        self.param.lang.objects = list(self._voices_by_language.keys())
        if 'en-US' in self._voices_by_language:
            default_lang = 'en-US'
        else:
            default_lang = list(self._voices_by_language.keys())[0]
        self.param.lang.default = default_lang
        self.lang = default_lang
        self.param.trigger('lang')

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