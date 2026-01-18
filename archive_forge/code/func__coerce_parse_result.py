import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pip._vendor.pyparsing import (  # noqa: N817
from .specifiers import InvalidSpecifier, Specifier
def _coerce_parse_result(results: Union[ParseResults, List[Any]]) -> List[Any]:
    if isinstance(results, ParseResults):
        return [_coerce_parse_result(i) for i in results]
    else:
        return results