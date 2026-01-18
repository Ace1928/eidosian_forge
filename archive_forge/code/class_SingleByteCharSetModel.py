from typing import Dict, List, NamedTuple, Optional, Union
from .charsetprober import CharSetProber
from .enums import CharacterCategory, ProbingState, SequenceLikelihood
class SingleByteCharSetModel(NamedTuple):
    charset_name: str
    language: str
    char_to_order_map: Dict[int, int]
    language_model: Dict[int, Dict[int, int]]
    typical_positive_ratio: float
    keep_ascii_letters: bool
    alphabet: str