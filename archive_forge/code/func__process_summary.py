import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_summary(self, value: str) -> str:
    """Check the field contains no newlines."""
    if '\n' in value:
        raise self._invalid_metadata('{field} must be a single line')
    return value