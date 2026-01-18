from __future__ import annotations
import logging # isort:skip
import re
from typing import Any
from ... import colors
from .. import enums
from .bases import Init, Property
from .container import Tuple
from .either import Either
from .enum import Enum
from .numeric import Byte, Percent
from .singletons import Undefined
from .string import Regex
 ref Color

    The only difference to Color is that this class transforms values into
    hexadecimal strings to be sent to BokehJS.

    