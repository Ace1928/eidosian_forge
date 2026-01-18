import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class UnusableRedirect(errors.BzrError):
    _fmt = 'Unable to follow redirect from %(source)s to %(target)s: %(reason)s.'

    def __init__(self, source, target, reason):
        super().__init__(source=source, target=target, reason=reason)