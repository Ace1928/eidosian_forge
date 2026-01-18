from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
class SpeechToText(Widget):
    """
    The SpeechToText widget controls the speech recognition service of
    the browser.

    It wraps the HTML5 SpeechRecognition API.  See
    https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition

    Reference: https://panel.holoviz.org/reference/widgets/SpeechToText.html

    :Example:

    >>> SpeechToText(button_type="light")

    This functionality is **experimental** and only supported by
    Chrome and a few other browsers.  Checkout
    https://caniuse.com/speech-recognition for a up to date list of
    browsers supporting the SpeechRecognition Api. Or alternatively
    https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition#Browser_compatibility

    On some browsers, like Chrome, using Speech Recognition on a web
    page involves a server-based recognition engine. Your audio is
    sent to a web service for recognition processing, so it won't work
    offline. Whether this is secure and confidential enough for your
    use case is up to you to evaluate.
    """
    abort = param.Event(doc="\n        Stops the speech recognition service from listening to\n        incoming audio, and doesn't attempt to return a\n        RecognitionResult.")
    start = param.Event(doc='\n        Starts the speech recognition service listening to incoming\n        audio with intent to recognize grammars associated with the\n        current SpeechRecognition.')
    stop = param.Event(doc='\n        Stops the speech recognition service from listening to\n        incoming audio, and attempts to return a RecognitionResult\n        using the audio captured so far.')
    lang = param.ObjectSelector(default='', objects=[''] + LANGUAGE_CODES, allow_None=True, label='Language', doc="\n        The language of the current SpeechRecognition in BCP 47\n        format. For example 'en-US'. If not specified, this defaults\n        to the HTML lang attribute value, or the user agent's language\n        setting if that isn't set either.  ")
    continuous = param.Boolean(default=False, doc='\n        Controls whether continuous results are returned for each\n        recognition, or only a single result. Defaults to False')
    interim_results = param.Boolean(default=False, doc='\n        Controls whether interim results should be returned (True) or\n        not (False.) Interim results are results that are not yet\n        final (e.g. the RecognitionResult.is_final property is\n        False).')
    max_alternatives = param.Integer(default=1, bounds=(1, 5), doc='\n        Sets the maximum number of RecognitionAlternatives provided\n        per result.  A number between 1 and 5. The default value is\n        1.')
    service_uri = param.String(doc="\n        Specifies the location of the speech recognition service used\n        by the current SpeechRecognition to handle the actual\n        recognition. The default is the user agent's default speech\n        service.")
    grammars = param.ClassSelector(class_=GrammarList, doc='\n        A GrammarList object that represents the grammars that will be\n        understood by the current SpeechRecognition service')
    button_hide = param.Boolean(default=False, label='Hide the Button', doc='\n        If True no button is shown. If False a toggle Start/ Stop button is shown.')
    button_type = param.ObjectSelector(default='light', objects=BUTTON_TYPES, doc='\n        The button styling.')
    button_not_started = param.String(label='Button Text when not started', doc="\n        The text to show on the button when the SpeechRecognition\n        service is NOT started.  If '' a *muted microphone* icon is\n        shown.")
    button_started = param.String(label='Button Text when started', doc="\n        The text to show on the button when the SpeechRecognition\n        service is started. If '' a *muted microphone* icon is\n        shown.")
    started = param.Boolean(constant=True, doc='\n        Returns True if the Speech Recognition Service is started and\n        False otherwise.')
    audio_started = param.Boolean(constant=True, doc='\n        Returns True if the Audio is started and False otherwise.')
    sound_started = param.Boolean(constant=True, doc='\n        Returns True if the Sound is started and False otherwise.')
    speech_started = param.Boolean(constant=True, doc='\n        Returns True if the the User has started speaking and False otherwise.')
    results = param.List(constant=True, doc='\n        The `results` as a list of Dictionaries.')
    value = param.String(constant=True, label='Last Result', doc='\n        The transcipt of the highest confidence RecognitionAlternative\n        of the last RecognitionResult. Please note we strip the\n        transcript for leading spaces.')
    _grammars = param.List(constant=True, doc='\n        List used to transfer the serialized grammars from server to\n        browser.')
    _rename: ClassVar[Mapping[str, str | None]] = {'grammars': None, '_grammars': 'grammars', 'name': None, 'value': None}
    _widget_type: ClassVar[Type[Model]] = _BkSpeechToText

    def __init__(self, **params):
        super().__init__(**params)
        if self.grammars:
            self._update_grammars()

    def __repr__(self, depth=None):
        return f"SpeechToText(name='{self.name}')"

    @param.depends('grammars', watch=True)
    def _update_grammars(self):
        with param.edit_constant(self):
            if self.grammars:
                self._grammars = self.grammars.serialize()
            else:
                self._grammars = []

    @param.depends('results', watch=True)
    def _update_results(self):
        with param.edit_constant(self):
            if self.results and 'alternatives' in self.results[-1]:
                self.value = self.results[-1]['alternatives'][0]['transcript'].lstrip()
            else:
                self.value = ''

    @property
    def results_deserialized(self):
        """
        Returns the results as a List of RecognitionResults
        """
        return RecognitionResult.create_from_list(self.results)

    @property
    def results_as_html(self) -> str:
        """
        Returns the `results` formatted as html

        Convenience method for ease of use
        """
        if not self.results:
            return 'No results'
        html = '<div class="pn-speech-recognition-result">'
        total = len(self.results) - 1
        for index, result in enumerate(reversed(self.results_deserialized)):
            if len(self.results) > 1:
                html += f'<h3>Result {total - index}</h3>'
            html += f'<span>Is Final: {result.is_final}</span><br/>'
            for index2, alternative in enumerate(result.alternatives):
                if len(result.alternatives) > 1:
                    html += f'<h4>Alternative {index2}</h4>'
                html += f'\n                <span>Confidence: {alternative.confidence:.2f}</span>\n                </br>\n                <p>\n                  <strong>{alternative.transcript}</strong>\n                </p>\n                '
        html += '</div>'
        return html