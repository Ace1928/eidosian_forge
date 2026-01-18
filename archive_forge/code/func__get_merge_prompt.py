import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
def _get_merge_prompt(self, prompt, to, subject, attachment):
    """See MailClient._get_merge_prompt"""
    return '%s\n\nTo: %s\nSubject: %s\n\n%s' % (prompt, to, subject, attachment.decode('utf-8', 'replace'))