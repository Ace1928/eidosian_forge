import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _handle_simple_dict(self, value: dict, path: Sequence[str]) -> Optional[str]:
    inline = any((p in value for p in self._guess_inline_defs))
    simple = not any((isinstance(v, (list, dict)) for v in value.values()))
    if inline or simple:
        return f'{{{', '.join(self._inline_attrs(value, path))}}}\n'
    return None