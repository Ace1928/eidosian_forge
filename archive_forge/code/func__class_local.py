import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
@classmethod
def _class_local(cls, name, default):
    """
        Returns an attribute directly associated with this class
        (as opposed to subclasses), setting default if necessary
        """
    result = cls.__dict__.get(name, default)
    setattr(cls, name, result)
    return result