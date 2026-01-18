from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def add_comments(self, comments: t.Optional[t.List[str]]) -> None:
    if self.comments is None:
        self.comments = []
    if comments:
        for comment in comments:
            _, *meta = comment.split(SQLGLOT_META)
            if meta:
                for kv in ''.join(meta).split(','):
                    k, *v = kv.split('=')
                    value = v[0].strip() if v else True
                    self.meta[k.strip()] = value
            self.comments.append(comment)