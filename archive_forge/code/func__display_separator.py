from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def _display_separator(self) -> str:
    """
        Get the symbol that should be used to separate multiple clarifying
        details -- such as a comma in English, or an ideographic comma in
        Japanese.

        Requires that `language_data` is installed.
        """
    try:
        from language_data.names import DISPLAY_SEPARATORS
    except ImportError:
        print(LANGUAGE_NAME_IMPORT_MESSAGE, file=sys.stdout)
        raise
    if self._disp_separator is not None:
        return self._disp_separator
    matched, _dist = closest_match(self, DISPLAY_SEPARATORS.keys())
    self._disp_separator = DISPLAY_SEPARATORS[matched]
    return self._disp_separator