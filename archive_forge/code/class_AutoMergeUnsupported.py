import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class AutoMergeUnsupported(errors.BzrError):
    _fmt = 'The merge proposal %(mp)s does not support automerge.'