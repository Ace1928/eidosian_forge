import re
from collections import OrderedDict
from typing import Any, Optional
def _casefold(value: str) -> str:
    try:
        return value.casefold()
    except AttributeError:
        return value.lower()