import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class Claws(ExternalMailClient):
    __doc__ = 'Claws mail client.'
    supports_body = True
    _client_commands = ['claws-mail']

    def _get_compose_commandline(self, to, subject, attach_path, body=None, from_=None):
        """See ExternalMailClient._get_compose_commandline"""
        compose_url = []
        if from_ is not None:
            compose_url.append('from=' + urlutils.quote(from_))
        if subject is not None:
            compose_url.append('subject=' + urlutils.quote(self._encode_safe(subject)))
        if body is not None:
            compose_url.append('body=' + urlutils.quote(self._encode_safe(body)))
        if to is None:
            raise NoMailAddressSpecified()
        compose_url = 'mailto:{}?{}'.format(self._encode_safe(to), '&'.join(compose_url))
        message_options = ['--compose', compose_url]
        if attach_path is not None:
            message_options.extend(['--attach', self._encode_path(attach_path, 'attachment')])
        return message_options

    def _compose(self, prompt, to, subject, attach_path, mime_subtype, extension, body=None, from_=None):
        """See ExternalMailClient._compose"""
        if from_ is None:
            from_ = self.config.get('email')
        super()._compose(prompt, to, subject, attach_path, mime_subtype, extension, body, from_)