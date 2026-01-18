import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
class SMTPError(BzrError):
    _fmt = 'SMTP error: %(error)s'

    def __init__(self, error):
        self.error = error