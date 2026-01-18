from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
@staticmethod
def _filter_keys(d: dict, keys: Iterable[str]) -> dict:
    """
        Select a subset of keys from a dictionary.
        """
    return {key: d[key] for key in keys if key in d}