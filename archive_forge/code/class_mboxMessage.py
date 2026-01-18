import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
class mboxMessage(_mboxMMDFMessage):
    """Message with mbox-specific properties."""