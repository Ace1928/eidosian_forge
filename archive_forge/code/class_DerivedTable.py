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
class DerivedTable(Expression):

    @property
    def selects(self) -> t.List[Expression]:
        return self.this.selects if isinstance(self.this, Query) else []

    @property
    def named_selects(self) -> t.List[str]:
        return [select.output_name for select in self.selects]