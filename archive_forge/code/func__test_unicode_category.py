import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def _test_unicode_category(s: str, categories: Sequence[str]) -> bool:
    if len(s) != 1:
        return all((_test_unicode_category(char, categories) for char in s))
    return s == '_' or unicodedata.category(s) in categories