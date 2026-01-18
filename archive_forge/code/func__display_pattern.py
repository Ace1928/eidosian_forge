from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def _display_pattern(self) -> str:
    """
        Get the pattern, according to CLDR, that should be used for clarifying
        details of a language code.
        """
    if self._disp_pattern is not None:
        return self._disp_pattern
    if self.distance(Language.get('zh')) <= 25:
        self._disp_pattern = '{0}（{1}）'
    else:
        self._disp_pattern = '{0} ({1})'
    return self._disp_pattern