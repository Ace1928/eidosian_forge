import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class MailClientNotFound(errors.BzrError):
    _fmt = 'Unable to find mail client with the following names: %(mail_command_list_string)s'

    def __init__(self, mail_command_list):
        mail_command_list_string = ', '.join(mail_command_list)
        errors.BzrError.__init__(self, mail_command_list=mail_command_list, mail_command_list_string=mail_command_list_string)