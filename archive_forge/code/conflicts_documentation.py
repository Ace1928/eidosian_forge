import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
Resolve the conflict by copying one of .THIS or .OTHER into file.

        :param tt: The TreeTransform where the conflict is resolved.
        :param winner_suffix: Either 'THIS' or 'OTHER'

        The resolution is symmetric, when taking THIS, item.THIS is renamed
        into item and vice-versa. This takes one of the files as a whole
        ignoring every difference that could have been merged cleanly.
        