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
class ColorHex(Color):
    """ ref Color

    The only difference to Color is that this class transforms values into
    hexadecimal strings to be sent to BokehJS.

    """

    def transform(self, value: Any) -> Any:
        if isinstance(value, str):
            value = value.lower()
            if value.startswith('rgb'):
                match = re.findall('[\\d\\.]+', value)
                a = float(match[3]) if value[3] == 'a' else 1.0
                value = colors.RGB(int(match[0]), int(match[1]), int(match[2]), a).to_hex()
            elif value in enums.NamedColor:
                value = getattr(colors.named, value).to_hex()
        elif isinstance(value, tuple):
            value = colors.RGB(*value).to_hex()
        else:
            value = value.to_hex()
        return value.lower()