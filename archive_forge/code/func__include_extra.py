import os
import stat
import textwrap
from email import message_from_file
from email.message import Message
from tempfile import NamedTemporaryFile
from typing import Optional, List
from distutils.util import rfc822_escape
from . import _normalization, _reqs
from .extern.packaging.markers import Marker
from .extern.packaging.requirements import Requirement
from .extern.packaging.version import Version
from .warnings import SetuptoolsDeprecationWarning
def _include_extra(req: str, extra: str, condition: str) -> Requirement:
    r = Requirement(req)
    parts = (f'({r.marker})' if r.marker else None, f'({condition})' if condition else None, f'extra == {extra!r}' if extra else None)
    r.marker = Marker(' and '.join((x for x in parts if x)))
    return r