import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def _describe_token_type(token_type: str) -> str:
    if token_type in reverse_operators:
        return reverse_operators[token_type]
    return {TOKEN_COMMENT_BEGIN: 'begin of comment', TOKEN_COMMENT_END: 'end of comment', TOKEN_COMMENT: 'comment', TOKEN_LINECOMMENT: 'comment', TOKEN_BLOCK_BEGIN: 'begin of statement block', TOKEN_BLOCK_END: 'end of statement block', TOKEN_VARIABLE_BEGIN: 'begin of print statement', TOKEN_VARIABLE_END: 'end of print statement', TOKEN_LINESTATEMENT_BEGIN: 'begin of line statement', TOKEN_LINESTATEMENT_END: 'end of line statement', TOKEN_DATA: 'template data / text', TOKEN_EOF: 'end of template'}.get(token_type, token_type)