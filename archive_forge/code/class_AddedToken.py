import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@dataclass(frozen=False, eq=True)
class AddedToken:
    """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.

        The `normalized` will default to `not special` if it is not specified, similarly to the definition in
        `tokenizers`.
        """

    def __init__(self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None):
        self.content = content
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.special = special
        self.normalized = normalized if normalized is not None else not special

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.content