import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
class _ErrorFormatting:

    def __init__(self, ex: JsonSchemaValueException):
        self.ex = ex
        self.name = f'`{self._simplify_name(ex.name)}`'
        self._original_message = self.ex.message.replace(ex.name, self.name)
        self._summary = ''
        self._details = ''

    def __str__(self) -> str:
        if _logger.getEffectiveLevel() <= logging.DEBUG and self.details:
            return f'{self.summary}\n\n{self.details}'
        return self.summary

    @property
    def summary(self) -> str:
        if not self._summary:
            self._summary = self._expand_summary()
        return self._summary

    @property
    def details(self) -> str:
        if not self._details:
            self._details = self._expand_details()
        return self._details

    def _simplify_name(self, name):
        x = len('data.')
        return name[x:] if name.startswith('data.') else name

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

    def _expand_details(self) -> str:
        optional = []
        desc_lines = self.ex.definition.pop('$$description', [])
        desc = self.ex.definition.pop('description', None) or ' '.join(desc_lines)
        if desc:
            description = '\n'.join(wrap(desc, width=80, initial_indent='    ', subsequent_indent='    ', break_long_words=False))
            optional.append(f'DESCRIPTION:\n{description}')
        schema = json.dumps(self.ex.definition, indent=4)
        value = json.dumps(self.ex.value, indent=4)
        defaults = [f'GIVEN VALUE:\n{indent(value, '    ')}', f'OFFENDING RULE: {self.ex.rule!r}', f'DEFINITION:\n{indent(schema, '    ')}']
        return '\n\n'.join(optional + defaults)