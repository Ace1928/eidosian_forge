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
def _stringTags(self, tags):
    """
        Converts a tag dictionary to a string.

        @param tags: The tag dict passed to sendMsg.

        @rtype: L{unicode}
        @return: IRCv3-format tag string
        """
    self._validateTags(tags)
    tagStrings = []
    for tag, value in tags.items():
        if value:
            tagStrings.append(f'{tag}={self._escapeTagValue(value)}')
        else:
            tagStrings.append(tag)
    return ';'.join(tagStrings)