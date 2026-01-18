import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
class CliDetectionResult:

    def __init__(self, path: str, encoding: Optional[str], encoding_aliases: List[str], alternative_encodings: List[str], language: str, alphabets: List[str], has_sig_or_bom: bool, chaos: float, coherence: float, unicode_path: Optional[str], is_preferred: bool):
        self.path = path
        self.unicode_path = unicode_path
        self.encoding = encoding
        self.encoding_aliases = encoding_aliases
        self.alternative_encodings = alternative_encodings
        self.language = language
        self.alphabets = alphabets
        self.has_sig_or_bom = has_sig_or_bom
        self.chaos = chaos
        self.coherence = coherence
        self.is_preferred = is_preferred

    @property
    def __dict__(self) -> Dict[str, Any]:
        return {'path': self.path, 'encoding': self.encoding, 'encoding_aliases': self.encoding_aliases, 'alternative_encodings': self.alternative_encodings, 'language': self.language, 'alphabets': self.alphabets, 'has_sig_or_bom': self.has_sig_or_bom, 'chaos': self.chaos, 'coherence': self.coherence, 'unicode_path': self.unicode_path, 'is_preferred': self.is_preferred}

    def to_json(self) -> str:
        return dumps(self.__dict__, ensure_ascii=True, indent=4)