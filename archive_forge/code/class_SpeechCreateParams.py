from __future__ import annotations
from typing import Union
from typing_extensions import Literal, Required, TypedDict
class SpeechCreateParams(TypedDict, total=False):
    input: Required[str]
    'The text to generate audio for. The maximum length is 4096 characters.'
    model: Required[Union[str, Literal['tts-1', 'tts-1-hd']]]
    '\n    One of the available [TTS models](https://platform.openai.com/docs/models/tts):\n    `tts-1` or `tts-1-hd`\n    '
    voice: Required[Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']]
    'The voice to use when generating the audio.\n\n    Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.\n    Previews of the voices are available in the\n    [Text to speech guide](https://platform.openai.com/docs/guides/text-to-speech/voice-options).\n    '
    response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']
    'The format to audio in.\n\n    Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.\n    '
    speed: float
    'The speed of the generated audio.\n\n    Select a value from `0.25` to `4.0`. `1.0` is the default.\n    '