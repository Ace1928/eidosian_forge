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
class _FormattingState(_textattributes._FormattingStateMixin):
    """
    Formatting state/attributes of a single character.

    Attributes include:
        - Formatting nullifier
        - Bold
        - Underline
        - Reverse video
        - Foreground color
        - Background color

    @since: 13.1
    """
    compareAttributes = ('off', 'bold', 'underline', 'reverseVideo', 'foreground', 'background')

    def __init__(self, off=False, bold=False, underline=False, reverseVideo=False, foreground=None, background=None):
        self.off = off
        self.bold = bold
        self.underline = underline
        self.reverseVideo = reverseVideo
        self.foreground = foreground
        self.background = background

    def toMIRCControlCodes(self):
        """
        Emit a mIRC control sequence that will set up all the attributes this
        formatting state has set.

        @return: A string containing mIRC control sequences that mimic this
            formatting state.
        """
        attrs = []
        if self.bold:
            attrs.append(_BOLD)
        if self.underline:
            attrs.append(_UNDERLINE)
        if self.reverseVideo:
            attrs.append(_REVERSE_VIDEO)
        if self.foreground is not None or self.background is not None:
            c = ''
            if self.foreground is not None:
                c += '%02d' % (self.foreground,)
            if self.background is not None:
                c += ',%02d' % (self.background,)
            attrs.append(_COLOR + c)
        return _OFF + ''.join(map(str, attrs))