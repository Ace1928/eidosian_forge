import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class MailClient:
    """A mail client that can send messages with attachements."""

    def __init__(self, config):
        self.config = config

    def compose(self, prompt, to, subject, attachment, mime_subtype, extension, basename=None, body=None):
        """Compose (and possibly send) an email message

        Must be implemented by subclasses.

        :param prompt: A message to tell the user what to do.  Supported by
            the Editor client, but ignored by others
        :param to: The address to send the message to
        :param subject: The contents of the subject line
        :param attachment: An email attachment, as a bytestring
        :param mime_subtype: The attachment is assumed to be a subtype of
            Text.  This allows the precise subtype to be specified, e.g.
            "plain", "x-patch", etc.
        :param extension: The file extension associated with the attachment
            type, e.g. ".patch"
        :param basename: The name to use for the attachment, e.g.
            "send-nick-3252"
        """
        raise NotImplementedError

    def compose_merge_request(self, to, subject, directive, basename=None, body=None):
        """Compose (and possibly send) a merge request

        :param to: The address to send the request to
        :param subject: The subject line to use for the request
        :param directive: A merge directive representing the merge request, as
            a bytestring.
        :param basename: The name to use for the attachment, e.g.
            "send-nick-3252"
        """
        prompt = self._get_merge_prompt('Please describe these changes:', to, subject, directive)
        self.compose(prompt, to, subject, directive, 'x-patch', '.patch', basename, body)

    def _get_merge_prompt(self, prompt, to, subject, attachment):
        """Generate a prompt string.  Overridden by Editor.

        :param prompt: A string suggesting what user should do
        :param to: The address the mail will be sent to
        :param subject: The subject line of the mail
        :param attachment: The attachment that will be used
        """
        return ''