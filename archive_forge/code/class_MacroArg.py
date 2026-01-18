import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
@attr.s(auto_attribs=True, frozen=True)
class MacroArg:
    """
    Information used to replace or unescape arguments in a macro value when the macro is resolved
    Normal argument syntax:    {5}
    Escaped argument syntax:  {{5}}
    """
    start_index: int = attr.ib(validator=attr.validators.instance_of(int))
    number_str: str = attr.ib(validator=attr.validators.instance_of(str))
    is_escaped: bool = attr.ib(validator=attr.validators.instance_of(bool))
    macro_normal_arg_pattern = re.compile('(?<!{){\\d+}|{\\d+}(?!})')
    macro_escaped_arg_pattern = re.compile('{{2}\\d+}{2}')
    digit_pattern = re.compile('\\d+')