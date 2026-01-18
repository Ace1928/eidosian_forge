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
class IndexConstraintOption(Expression):
    arg_types = {'key_block_size': False, 'using': False, 'parser': False, 'comment': False, 'visible': False, 'engine_attr': False, 'secondary_engine_attr': False}