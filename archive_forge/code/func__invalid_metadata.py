import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _invalid_metadata(self, msg: str, cause: Optional[Exception]=None) -> InvalidMetadata:
    exc = InvalidMetadata(self.raw_name, msg.format_map({'field': repr(self.raw_name)}))
    exc.__cause__ = cause
    return exc