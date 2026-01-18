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
class LoadData(Expression):
    arg_types = {'this': True, 'local': False, 'overwrite': False, 'inpath': True, 'partition': False, 'input_format': False, 'serde': False}