from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def _best_name(self, names: Mapping[str, str], language: 'Language', max_distance: int):
    matchable_languages = set(language.broader_tags())
    possible_languages = [key for key in sorted(names.keys()) if key in matchable_languages]
    target_language, score = closest_match(language, possible_languages, max_distance)
    if target_language in names:
        return names[target_language]
    else:
        return names.get(DEFAULT_LANGUAGE)