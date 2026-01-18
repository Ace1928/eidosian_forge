import ast
import dataclasses
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
Evaluate the match expression.

        :param matcher:
            Given an identifier, should return whether it matches or not.
            Should be prepared to handle arbitrary strings as input.

        :returns: Whether the expression matches or not.
        