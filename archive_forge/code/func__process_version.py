import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_version(self, value: str) -> version_module.Version:
    if not value:
        raise self._invalid_metadata('{field} is a required field')
    try:
        return version_module.parse(value)
    except version_module.InvalidVersion as exc:
        raise self._invalid_metadata(f'{value!r} is invalid for {{field}}', cause=exc)