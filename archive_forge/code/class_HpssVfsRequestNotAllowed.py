import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class HpssVfsRequestNotAllowed(errors.BzrError):
    _fmt = 'VFS requests over the smart server are not allowed. Encountered: %(method)s, %(arguments)s.'

    def __init__(self, method, arguments):
        self.method = method
        self.arguments = arguments