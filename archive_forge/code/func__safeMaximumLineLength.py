import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def _safeMaximumLineLength(self, command):
    """
        Estimate a safe maximum line length for the given command.

        This is done by assuming the maximum values for nickname length,
        realname and hostname combined with the command that needs to be sent
        and some guessing. A theoretical maximum value is used because it is
        possible that our nickname, username or hostname changes (on the server
        side) while the length is still being calculated.
        """
    theoretical = ':{}!{}@{} {}'.format('a' * self.supported.getFeature('NICKLEN'), 'b' * 10, 'c' * 63, command)
    fudge = 10
    return MAX_COMMAND_LENGTH - len(theoretical) - fudge