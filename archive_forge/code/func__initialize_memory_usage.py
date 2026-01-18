from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _initialize_memory_usage(memory_usage: bool | str | None=None) -> bool | str:
    """Get memory usage based on inputs and display options."""
    if memory_usage is None:
        memory_usage = get_option('display.memory_usage')
    return memory_usage