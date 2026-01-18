from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def accept_suggestion(input_text: str, auto_accept: bool) -> bool:
    sys.stdout.write(input_text)
    if auto_accept:
        sys.stdout.write('Y\n')
        return True
    return input().lower() != 'n'