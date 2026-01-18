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
def _read_field_unescaped_from_msg(msg: Message, field: str) -> Optional[str]:
    """Read Message header field and apply rfc822_unescape."""
    value = _read_field_from_msg(msg, field)
    if value is None:
        return value
    return rfc822_unescape(value)