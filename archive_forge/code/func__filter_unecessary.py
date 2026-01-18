import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _filter_unecessary(self, schema: dict, path: Sequence[str]):
    return {key: value for key, value in schema.items() if not self._is_unecessary([*path, key])}