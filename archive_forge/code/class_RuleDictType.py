from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
class RuleDictType(_RuleDictReqType, total=False):
    pre: Any
    post: Any