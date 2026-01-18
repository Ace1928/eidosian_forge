from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _show_tag_conficts(self, to_file):
    if not getattr(self, 'tag_conflicts', None):
        return
    to_file.write('Conflicting tags:\n')
    for name, value1, value2 in self.tag_conflicts:
        to_file.write('    {}\n'.format(name))