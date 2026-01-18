import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_requires_dist(self, value: List[str]) -> List[requirements.Requirement]:
    reqs = []
    try:
        for req in value:
            reqs.append(requirements.Requirement(req))
    except requirements.InvalidRequirement as exc:
        raise self._invalid_metadata(f'{req!r} is invalid for {{field}}', cause=exc)
    else:
        return reqs