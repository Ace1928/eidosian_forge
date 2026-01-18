import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
def _expand_summary(self):
    msg = self._original_message
    for bad, repl in _MESSAGE_REPLACEMENTS.items():
        msg = msg.replace(bad, repl)
    if any((substring in msg for substring in _SKIP_DETAILS)):
        return msg
    schema = self.ex.rule_definition
    if self.ex.rule in _NEED_DETAILS and schema:
        summary = _SummaryWriter(_TOML_JARGON)
        return f'{msg}:\n\n{indent(summary(schema), '    ')}'
    return msg