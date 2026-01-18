from __future__ import annotations
import re
from typing import Any, List, Optional
from langchain_text_splitters.base import Language, TextSplitter
def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f'({separator})', text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != '']