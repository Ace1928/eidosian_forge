import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
class SignatureVerificationFailed(errors.BzrError):
    _fmt = 'Failed to verify GPG signature data with error "%(error)s"'

    def __init__(self, error):
        errors.BzrError.__init__(self, error=error)