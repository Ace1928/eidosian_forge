import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
@dataclasses.dataclass
class UnicodeLatexTranslation:
    unicode: str
    latex: str
    encode: bool
    decode: bool
    text_mode: bool
    math_mode: bool