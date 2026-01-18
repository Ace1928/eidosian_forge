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
class DateTrunc(Func):
    arg_types = {'unit': True, 'this': True, 'zone': False}

    def __init__(self, **args):
        unit = args.get('unit')
        if isinstance(unit, TimeUnit.VAR_LIKE):
            args['unit'] = Literal.string((TimeUnit.UNABBREVIATED_UNIT_NAME.get(unit.name) or unit.name).upper())
        elif isinstance(unit, Week):
            unit.set('this', Literal.string(unit.this.name.upper()))
        super().__init__(**args)

    @property
    def unit(self) -> Expression:
        return self.args['unit']