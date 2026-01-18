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
class IndexColumnConstraint(ColumnConstraintKind):
    arg_types = {'this': False, 'expressions': False, 'kind': False, 'index_type': False, 'options': False, 'expression': False, 'granularity': False}