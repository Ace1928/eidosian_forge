import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
Show warning for errors seen by rmtree.
        