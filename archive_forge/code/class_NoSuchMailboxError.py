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
class NoSuchMailboxError(Error):
    """The specified mailbox does not exist and won't be created."""