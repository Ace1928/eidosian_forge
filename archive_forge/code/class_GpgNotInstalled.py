import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
class GpgNotInstalled(errors.DependencyNotPresent):
    _fmt = 'python-gpg is not installed, it is needed to create or verify signatures. %(error)s'

    def __init__(self, error):
        errors.DependencyNotPresent.__init__(self, 'gpg', error)