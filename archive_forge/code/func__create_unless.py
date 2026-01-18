from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
def _create_unless(terminals, g_regex_flags, re_, use_bytes):
    tokens_by_type = classify(terminals, lambda t: type(t.pattern))
    assert len(tokens_by_type) <= 2, tokens_by_type.keys()
    embedded_strs = set()
    callback = {}
    for retok in tokens_by_type.get(PatternRE, []):
        unless = []
        for strtok in tokens_by_type.get(PatternStr, []):
            if strtok.priority != retok.priority:
                continue
            s = strtok.pattern.value
            if s == _get_match(re_, retok.pattern.to_regexp(), s, g_regex_flags):
                unless.append(strtok)
                if strtok.pattern.flags <= retok.pattern.flags:
                    embedded_strs.add(strtok)
        if unless:
            callback[retok.name] = UnlessCallback(Scanner(unless, g_regex_flags, re_, match_whole=True, use_bytes=use_bytes))
    new_terminals = [t for t in terminals if t not in embedded_strs]
    return (new_terminals, callback)