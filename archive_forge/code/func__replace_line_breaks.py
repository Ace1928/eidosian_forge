from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _replace_line_breaks(self, string: str) -> str:
    """We don't want to extra indent line breaks so we temporarily replace them with sentinels."""
    if self.pretty:
        return string.replace('\n', self.SENTINEL_LINE_BREAK)
    return string