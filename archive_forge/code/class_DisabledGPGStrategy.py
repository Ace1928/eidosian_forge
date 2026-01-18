import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
class DisabledGPGStrategy:
    """A GPG Strategy that makes everything fail."""

    @staticmethod
    def verify_signatures_available():
        return True

    def __init__(self, ignored):
        """Real strategies take a configuration."""

    def sign(self, content, mode):
        raise SigningFailed('Signing is disabled.')

    def verify(self, signed_data, signature=None):
        raise SignatureVerificationFailed('Signature verification is disabled.')

    def set_acceptable_keys(self, command_line_input):
        pass