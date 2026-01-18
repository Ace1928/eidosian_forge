import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
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