import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class KMail(ExternalMailClient):
    __doc__ = 'KDE mail client.'
    _client_commands = ['kmail']

    def _get_compose_commandline(self, to, subject, attach_path):
        """See ExternalMailClient._get_compose_commandline"""
        message_options = []
        if subject is not None:
            message_options.extend(['-s', self._encode_safe(subject)])
        if attach_path is not None:
            message_options.extend(['--attach', self._encode_path(attach_path, 'attachment')])
        if to is not None:
            message_options.extend([self._encode_safe(to)])
        return message_options