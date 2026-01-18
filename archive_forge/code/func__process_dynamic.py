import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_dynamic(self, value: List[str]) -> List[str]:
    for dynamic_field in map(str.lower, value):
        if dynamic_field in {'name', 'version', 'metadata-version'}:
            raise self._invalid_metadata(f'{value!r} is not allowed as a dynamic field')
        elif dynamic_field not in _EMAIL_TO_RAW_MAPPING:
            raise self._invalid_metadata(f'{value!r} is not a valid dynamic field')
    return list(map(str.lower, value))