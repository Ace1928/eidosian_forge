from __future__ import annotations
import binascii
import datetime
import json
import os
import re
import sys
import typing as t
import uuid
from dataclasses import asdict, dataclass
from http.cookies import Morsel
from tornado import escape, httputil, web
from traitlets import Bool, Dict, Type, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_server.transutils import _i18n
from .security import passwd_check, set_password
from .utils import get_anonymous_username
@default('token')
def _token_default(self):
    if os.getenv('JUPYTER_TOKEN'):
        self.token_generated = False
        return os.environ['JUPYTER_TOKEN']
    if os.getenv('JUPYTER_TOKEN_FILE'):
        self.token_generated = False
        with open(os.environ['JUPYTER_TOKEN_FILE']) as token_file:
            return token_file.read()
    if not self.need_token:
        self.token_generated = False
        return ''
    else:
        self.token_generated = True
        return binascii.hexlify(os.urandom(24)).decode('ascii')