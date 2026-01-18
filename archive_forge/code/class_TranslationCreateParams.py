from __future__ import annotations
from typing import Union
from typing_extensions import Literal, Required, TypedDict
from ..._types import FileTypes
class TranslationCreateParams(TypedDict, total=False):
    file: Required[FileTypes]
    '\n    The audio file object (not file name) translate, in one of these formats: flac,\n    mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.\n    '
    model: Required[Union[str, Literal['whisper-1']]]
    'ID of the model to use.\n\n    Only `whisper-1` (which is powered by our open source Whisper V2 model) is\n    currently available.\n    '
    prompt: str
    "An optional text to guide the model's style or continue a previous audio\n    segment.\n\n    The [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)\n    should be in English.\n    "
    response_format: str
    '\n    The format of the transcript output, in one of these options: `json`, `text`,\n    `srt`, `verbose_json`, or `vtt`.\n    '
    temperature: float
    'The sampling temperature, between 0 and 1.\n\n    Higher values like 0.8 will make the output more random, while lower values like\n    0.2 will make it more focused and deterministic. If set to 0, the model will use\n    [log probability](https://en.wikipedia.org/wiki/Log_probability) to\n    automatically increase the temperature until certain thresholds are hit.\n    '