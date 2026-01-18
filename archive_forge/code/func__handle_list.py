import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _handle_list(self, schemas: list, prefix: str='', path: Sequence[str]=()) -> str:
    if self._is_unecessary(path):
        return ''
    repr_ = repr(schemas)
    if all((not isinstance(e, (dict, list)) for e in schemas)) and len(repr_) < 60:
        return f'{repr_}\n'
    item_prefix = self._child_prefix(prefix, '- ')
    return ''.join((self(v, item_prefix, _path=[*path, f'[{i}]']) for i, v in enumerate(schemas)))