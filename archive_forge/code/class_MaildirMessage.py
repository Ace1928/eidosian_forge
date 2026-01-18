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
class MaildirMessage(Message):
    """Message with Maildir-specific properties."""
    _type_specific_attributes = ['_subdir', '_info', '_date']

    def __init__(self, message=None):
        """Initialize a MaildirMessage instance."""
        self._subdir = 'new'
        self._info = ''
        self._date = time.time()
        Message.__init__(self, message)

    def get_subdir(self):
        """Return 'new' or 'cur'."""
        return self._subdir

    def set_subdir(self, subdir):
        """Set subdir to 'new' or 'cur'."""
        if subdir == 'new' or subdir == 'cur':
            self._subdir = subdir
        else:
            raise ValueError("subdir must be 'new' or 'cur': %s" % subdir)

    def get_flags(self):
        """Return as a string the flags that are set."""
        if self._info.startswith('2,'):
            return self._info[2:]
        else:
            return ''

    def set_flags(self, flags):
        """Set the given flags and unset all others."""
        self._info = '2,' + ''.join(sorted(flags))

    def add_flag(self, flag):
        """Set the given flag(s) without changing others."""
        self.set_flags(''.join(set(self.get_flags()) | set(flag)))

    def remove_flag(self, flag):
        """Unset the given string flag(s) without changing others."""
        if self.get_flags():
            self.set_flags(''.join(set(self.get_flags()) - set(flag)))

    def get_date(self):
        """Return delivery date of message, in seconds since the epoch."""
        return self._date

    def set_date(self, date):
        """Set delivery date of message, in seconds since the epoch."""
        try:
            self._date = float(date)
        except ValueError:
            raise TypeError("can't convert to float: %s" % date) from None

    def get_info(self):
        """Get the message's "info" as a string."""
        return self._info

    def set_info(self, info):
        """Set the message's "info" string."""
        if isinstance(info, str):
            self._info = info
        else:
            raise TypeError('info must be a string: %s' % type(info))

    def _explain_to(self, message):
        """Copy Maildir-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            message.set_flags(self.get_flags())
            message.set_subdir(self.get_subdir())
            message.set_date(self.get_date())
        elif isinstance(message, _mboxMMDFMessage):
            flags = set(self.get_flags())
            if 'S' in flags:
                message.add_flag('R')
            if self.get_subdir() == 'cur':
                message.add_flag('O')
            if 'T' in flags:
                message.add_flag('D')
            if 'F' in flags:
                message.add_flag('F')
            if 'R' in flags:
                message.add_flag('A')
            message.set_from('MAILER-DAEMON', time.gmtime(self.get_date()))
        elif isinstance(message, MHMessage):
            flags = set(self.get_flags())
            if 'S' not in flags:
                message.add_sequence('unseen')
            if 'R' in flags:
                message.add_sequence('replied')
            if 'F' in flags:
                message.add_sequence('flagged')
        elif isinstance(message, BabylMessage):
            flags = set(self.get_flags())
            if 'S' not in flags:
                message.add_label('unseen')
            if 'T' in flags:
                message.add_label('deleted')
            if 'R' in flags:
                message.add_label('answered')
            if 'P' in flags:
                message.add_label('forwarded')
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' % type(message))