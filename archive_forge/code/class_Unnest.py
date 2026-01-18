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
class Unnest(UDTF):
    arg_types = {'expressions': True, 'alias': False, 'offset': False}

    @property
    def selects(self) -> t.List[Expression]:
        columns = super().selects
        offset = self.args.get('offset')
        if offset:
            columns = columns + [to_identifier('offset') if offset is True else offset]
        return columns