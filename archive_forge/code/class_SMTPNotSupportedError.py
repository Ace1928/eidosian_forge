import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
class SMTPNotSupportedError(SMTPException):
    """The command or option is not supported by the SMTP server.

    This exception is raised when an attempt is made to run a command or a
    command with an option which is not supported by the server.
    """