from __future__ import annotations
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from operator import methodcaller
from typing import TYPE_CHECKING
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import urlopen
from warnings import warn
import contextlib
import json
import reprlib
import warnings
from attrs import define, field, fields
from jsonschema_specifications import REGISTRY as SPECIFICATIONS
from rpds import HashTrieMap
import referencing.exceptions
import referencing.jsonschema
from jsonschema import (
def _match_keyword(keyword):

    def matcher(value):
        if keyword in value:
            yield value
    return matcher