from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
class StringDistance(str, Enum):
    """Distance metric to use.

    Attributes:
        DAMERAU_LEVENSHTEIN: The Damerau-Levenshtein distance.
        LEVENSHTEIN: The Levenshtein distance.
        JARO: The Jaro distance.
        JARO_WINKLER: The Jaro-Winkler distance.
        HAMMING: The Hamming distance.
        INDEL: The Indel distance.
    """
    DAMERAU_LEVENSHTEIN = 'damerau_levenshtein'
    LEVENSHTEIN = 'levenshtein'
    JARO = 'jaro'
    JARO_WINKLER = 'jaro_winkler'
    HAMMING = 'hamming'
    INDEL = 'indel'