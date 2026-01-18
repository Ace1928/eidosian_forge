import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def _set_gpg_tty():
    tty = os.environ.get('TTY')
    if tty is not None:
        os.environ['GPG_TTY'] = tty
        trace.mutter('setting GPG_TTY=%s', tty)
    else:
        trace.mutter('** Env var TTY empty, cannot set GPG_TTY.  Is TTY exported?')