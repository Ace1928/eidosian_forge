import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class Evolution(BodyExternalMailClient):
    __doc__ = 'Evolution mail client.'
    _client_commands = ['evolution']

    def _get_compose_commandline(self, to, subject, attach_path, body=None):
        """See ExternalMailClient._get_compose_commandline"""
        message_options = {}
        if subject is not None:
            message_options['subject'] = subject
        if attach_path is not None:
            message_options['attach'] = attach_path
        if body is not None:
            message_options['body'] = body
        options_list = ['{}={}'.format(k, urlutils.escape(v)) for k, v in sorted(message_options.items())]
        return ['mailto:{}?{}'.format(self._encode_safe(to or ''), '&'.join(options_list))]