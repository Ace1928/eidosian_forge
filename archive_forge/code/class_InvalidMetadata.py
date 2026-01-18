import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
class InvalidMetadata(ValueError):
    """A metadata field contains invalid data."""
    field: str
    'The name of the field that contains invalid data.'

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)