import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def _field_over_url(field_idx: List[Tuple[int, int]], url_idx: List[Tuple[int, int]]):
    """Remove URL indices that overlap with filed list indices.

    Parameters
    ----------
    field_idx : list
        The list of field list index tuples.
    url_idx : list
        The list of URL index tuples.

    Returns
    -------
    url_idx : list
        The url_idx list with any tuples that have indices overlapping with field
        list indices removed.
    """
    for _fieldl, _fieldu in field_idx:
        for _key, _value in enumerate(url_idx):
            if _value[0] == _fieldl or _value[0] == _fieldu or _value[1] == _fieldl or (_value[1] == _fieldu):
                url_idx.pop(_key)
    return url_idx