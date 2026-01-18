import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
@classmethod
def _find_format(klass, registry, kind, format_string):
    try:
        first_line = format_string[:format_string.index(b'\n') + 1]
    except ValueError:
        first_line = format_string
    try:
        cls = registry.get(first_line)
    except KeyError:
        raise errors.UnknownFormatError(format=first_line, kind=kind)
    return cls.from_string(format_string)