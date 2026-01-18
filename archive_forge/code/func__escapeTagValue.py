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
def _escapeTagValue(self, value):
    """
        Escape the given tag value according to U{escaping rules in IRCv3
        <https://ircv3.net/specs/core/message-tags-3.2.html>}.

        @param value: The string value to escape.
        @type value: L{str}

        @return: The escaped string for sending as a message value
        @rtype: L{str}
        """
    return value.replace('\\', '\\\\').replace(';', '\\:').replace(' ', '\\s').replace('\r', '\\r').replace('\n', '\\n')