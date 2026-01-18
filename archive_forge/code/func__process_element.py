from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _process_element(element: Union[Tag, NavigableString, Comment], elements_to_skip: List[str], newline_elements: List[str]) -> str:
    """
    Traverse through HTML tree recursively to preserve newline and skip
    unwanted (code/binary) elements
    """
    from bs4 import NavigableString
    from bs4.element import Comment, Tag
    tag_name = getattr(element, 'name', None)
    if isinstance(element, Comment) or tag_name in elements_to_skip:
        return ''
    elif isinstance(element, NavigableString):
        return element
    elif tag_name == 'br':
        return '\n'
    elif tag_name in newline_elements:
        return ''.join((_process_element(child, elements_to_skip, newline_elements) for child in element.children if isinstance(child, (Tag, NavigableString, Comment)))) + '\n'
    else:
        return ''.join((_process_element(child, elements_to_skip, newline_elements) for child in element.children if isinstance(child, (Tag, NavigableString, Comment))))