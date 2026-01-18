from __future__ import annotations
from typing import List, Union
from typing_extensions import Literal, Required, TypedDict
from ..._types import FileTypes
class TranscriptionCreateParams(TypedDict, total=False):
    file: Required[FileTypes]
    '\n    The audio file object (not file name) to transcribe, in one of these formats:\n    flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.\n    '
    model: Required[Union[str, Literal['whisper-1']]]
    'ID of the model to use.\n\n    Only `whisper-1` (which is powered by our open source Whisper V2 model) is\n    currently available.\n    '
    language: str
    'The language of the input audio.\n\n    Supplying the input language in\n    [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will\n    improve accuracy and latency.\n    '
    prompt: str
    "An optional text to guide the model's style or continue a previous audio\n    segment.\n\n    The [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)\n    should match the audio language.\n    "
    response_format: Literal['json', 'text', 'srt', 'verbose_json', 'vtt']
    '\n    The format of the transcript output, in one of these options: `json`, `text`,\n    `srt`, `verbose_json`, or `vtt`.\n    '
    temperature: float
    'The sampling temperature, between 0 and 1.\n\n    Higher values like 0.8 will make the output more random, while lower values like\n    0.2 will make it more focused and deterministic. If set to 0, the model will use\n    [log probability](https://en.wikipedia.org/wiki/Log_probability) to\n    automatically increase the temperature until certain thresholds are hit.\n    '
    timestamp_granularities: List[Literal['word', 'segment']]
    'The timestamp granularities to populate for this transcription.\n\n    `response_format` must be set `verbose_json` to use timestamp granularities.\n    Either or both of these options are supported: `word`, or `segment`. Note: There\n    is no additional latency for segment timestamps, but generating word timestamps\n    incurs additional latency.\n    '