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
class ApproxDistinct(AggFunc):
    arg_types = {'this': True, 'accuracy': False}
    _sql_names = ['APPROX_DISTINCT', 'APPROX_COUNT_DISTINCT']