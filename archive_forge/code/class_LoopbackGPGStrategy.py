import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
class LoopbackGPGStrategy:
    """A GPG Strategy that acts like 'cat' - data is just passed through.
    Used in tests.
    """

    @staticmethod
    def verify_signatures_available():
        return True

    def __init__(self, ignored):
        """Real strategies take a configuration."""

    def sign(self, content, mode):
        return b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + content + b'-----END PSEUDO-SIGNED CONTENT-----\n'

    def verify(self, signed_data, signature=None):
        plain_text = signed_data.replace(b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n', b'')
        plain_text = plain_text.replace(b'-----END PSEUDO-SIGNED CONTENT-----\n', b'')
        return (SIGNATURE_VALID, None, plain_text)

    def set_acceptable_keys(self, command_line_input):
        if command_line_input is not None:
            patterns = command_line_input.split(',')
            self.acceptable_keys = []
            for pattern in patterns:
                if pattern == 'unknown':
                    pass
                else:
                    self.acceptable_keys.append(pattern)