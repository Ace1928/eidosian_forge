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
class _FormattingParser(_CommandDispatcherMixin):
    """
    A finite-state machine that parses formatted IRC text.

    Currently handled formatting includes: bold, reverse, underline,
    mIRC color codes and the ability to remove all current formatting.

    @see: U{http://www.mirc.co.uk/help/color.txt}

    @type _formatCodes: C{dict} mapping C{str} to C{str}
    @cvar _formatCodes: Mapping of format code values to names.

    @type state: C{str}
    @ivar state: Current state of the finite-state machine.

    @type _buffer: C{str}
    @ivar _buffer: Buffer, containing the text content, of the formatting
        sequence currently being parsed, the buffer is used as the content for
        L{_attrs} before being added to L{_result} and emptied upon calling
        L{emit}.

    @type _attrs: C{set}
    @ivar _attrs: Set of the applicable formatting states (bold, underline,
        etc.) for the current L{_buffer}, these are applied to L{_buffer} when
        calling L{emit}.

    @type foreground: L{_ForegroundColorAttr}
    @ivar foreground: Current foreground color attribute, or L{None}.

    @type background: L{_BackgroundColorAttr}
    @ivar background: Current background color attribute, or L{None}.

    @ivar _result: Current parse result.
    """
    prefix = 'state'
    _formatCodes = {_OFF: 'off', _BOLD: 'bold', _COLOR: 'color', _REVERSE_VIDEO: 'reverseVideo', _UNDERLINE: 'underline'}

    def __init__(self):
        self.state = 'TEXT'
        self._buffer = ''
        self._attrs = set()
        self._result = None
        self.foreground = None
        self.background = None

    def process(self, ch):
        """
        Handle input.

        @type ch: C{str}
        @param ch: A single character of input to process
        """
        self.dispatch(self.state, ch)

    def complete(self):
        """
        Flush the current buffer and return the final parsed result.

        @return: Structured text and attributes.
        """
        self.emit()
        if self._result is None:
            self._result = attributes.normal
        return self._result

    def emit(self):
        """
        Add the currently parsed input to the result.
        """
        if self._buffer:
            attrs = [getattr(attributes, name) for name in self._attrs]
            attrs.extend(filter(None, [self.foreground, self.background]))
            if not attrs:
                attrs.append(attributes.normal)
            attrs.append(self._buffer)
            attr = _foldr(operator.getitem, attrs.pop(), attrs)
            if self._result is None:
                self._result = attr
            else:
                self._result[attr]
            self._buffer = ''

    def state_TEXT(self, ch):
        """
        Handle the "text" state.

        Along with regular text, single token formatting codes are handled
        in this state too.

        @param ch: The character being processed.
        """
        formatName = self._formatCodes.get(ch)
        if formatName == 'color':
            self.emit()
            self.state = 'COLOR_FOREGROUND'
        elif formatName is None:
            self._buffer += ch
        else:
            self.emit()
            if formatName == 'off':
                self._attrs = set()
                self.foreground = self.background = None
            else:
                self._attrs.symmetric_difference_update([formatName])

    def state_COLOR_FOREGROUND(self, ch):
        """
        Handle the foreground color state.

        Foreground colors can consist of up to two digits and may optionally
        end in a I{,}. Any non-digit or non-comma characters are treated as
        invalid input and result in the state being reset to "text".

        @param ch: The character being processed.
        """
        if ch.isdigit() and len(self._buffer) < 2:
            self._buffer += ch
        else:
            if self._buffer:
                col = int(self._buffer) % len(_IRC_COLORS)
                self.foreground = getattr(attributes.fg, _IRC_COLOR_NAMES[col])
            else:
                self.foreground = self.background = None
            if ch == ',' and self._buffer:
                self._buffer = ''
                self.state = 'COLOR_BACKGROUND'
            else:
                self._buffer = ''
                self.state = 'TEXT'
                self.emit()
                self.process(ch)

    def state_COLOR_BACKGROUND(self, ch):
        """
        Handle the background color state.

        Background colors can consist of up to two digits and must occur after
        a foreground color and must be preceded by a I{,}. Any non-digit
        character is treated as invalid input and results in the state being
        set to "text".

        @param ch: The character being processed.
        """
        if ch.isdigit() and len(self._buffer) < 2:
            self._buffer += ch
        else:
            if self._buffer:
                col = int(self._buffer) % len(_IRC_COLORS)
                self.background = getattr(attributes.bg, _IRC_COLOR_NAMES[col])
                self._buffer = ''
            self.emit()
            self.state = 'TEXT'
            self.process(ch)