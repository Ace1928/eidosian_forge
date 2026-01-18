import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _is_unecessary(self, path: Sequence[str]) -> bool:
    if self._is_property(path) or not path:
        return False
    key = path[-1]
    return any((key.startswith(k) for k in '$_')) or key in self._IGNORE