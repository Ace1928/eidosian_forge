from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_link_ratio(section: Tag) -> float:
    links = section.find_all('a')
    total_text = ''.join((str(s) for s in section.stripped_strings))
    if len(total_text) == 0:
        return 0
    link_text = ''.join((str(string.string.strip()) for link in links for string in link.strings if string))
    return len(link_text) / len(total_text)