import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_provides_extra(self, value: List[str]) -> List[utils.NormalizedName]:
    normalized_names = []
    try:
        for name in value:
            normalized_names.append(utils.canonicalize_name(name, validate=True))
    except utils.InvalidName as exc:
        raise self._invalid_metadata(f'{name!r} is invalid for {{field}}', cause=exc)
    else:
        return normalized_names