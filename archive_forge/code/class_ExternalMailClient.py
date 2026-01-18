import errno
import os
import subprocess
import sys
import tempfile
from typing import Type
import breezy
from . import config as _mod_config
from . import email_message, errors, msgeditor, osutils, registry, urlutils
class ExternalMailClient(BodyExternalMailClient):
    __doc__ = 'An external mail client.'
    supports_body = False