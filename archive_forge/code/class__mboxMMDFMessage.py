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
class _mboxMMDFMessage(Message):
    """Message with mbox- or MMDF-specific properties."""
    _type_specific_attributes = ['_from']

    def __init__(self, message=None):
        """Initialize an mboxMMDFMessage instance."""
        self.set_from('MAILER-DAEMON', True)
        if isinstance(message, email.message.Message):
            unixfrom = message.get_unixfrom()
            if unixfrom is not None and unixfrom.startswith('From '):
                self.set_from(unixfrom[5:])
        Message.__init__(self, message)

    def get_from(self):
        """Return contents of "From " line."""
        return self._from

    def set_from(self, from_, time_=None):
        """Set "From " line, formatting and appending time_ if specified."""
        if time_ is not None:
            if time_ is True:
                time_ = time.gmtime()
            from_ += ' ' + time.asctime(time_)
        self._from = from_

    def get_flags(self):
        """Return as a string the flags that are set."""
        return self.get('Status', '') + self.get('X-Status', '')

    def set_flags(self, flags):
        """Set the given flags and unset all others."""
        flags = set(flags)
        status_flags, xstatus_flags = ('', '')
        for flag in ('R', 'O'):
            if flag in flags:
                status_flags += flag
                flags.remove(flag)
        for flag in ('D', 'F', 'A'):
            if flag in flags:
                xstatus_flags += flag
                flags.remove(flag)
        xstatus_flags += ''.join(sorted(flags))
        try:
            self.replace_header('Status', status_flags)
        except KeyError:
            self.add_header('Status', status_flags)
        try:
            self.replace_header('X-Status', xstatus_flags)
        except KeyError:
            self.add_header('X-Status', xstatus_flags)

    def add_flag(self, flag):
        """Set the given flag(s) without changing others."""
        self.set_flags(''.join(set(self.get_flags()) | set(flag)))

    def remove_flag(self, flag):
        """Unset the given string flag(s) without changing others."""
        if 'Status' in self or 'X-Status' in self:
            self.set_flags(''.join(set(self.get_flags()) - set(flag)))

    def _explain_to(self, message):
        """Copy mbox- or MMDF-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            flags = set(self.get_flags())
            if 'O' in flags:
                message.set_subdir('cur')
            if 'F' in flags:
                message.add_flag('F')
            if 'A' in flags:
                message.add_flag('R')
            if 'R' in flags:
                message.add_flag('S')
            if 'D' in flags:
                message.add_flag('T')
            del message['status']
            del message['x-status']
            maybe_date = ' '.join(self.get_from().split()[-5:])
            try:
                message.set_date(calendar.timegm(time.strptime(maybe_date, '%a %b %d %H:%M:%S %Y')))
            except (ValueError, OverflowError):
                pass
        elif isinstance(message, _mboxMMDFMessage):
            message.set_flags(self.get_flags())
            message.set_from(self.get_from())
        elif isinstance(message, MHMessage):
            flags = set(self.get_flags())
            if 'R' not in flags:
                message.add_sequence('unseen')
            if 'A' in flags:
                message.add_sequence('replied')
            if 'F' in flags:
                message.add_sequence('flagged')
            del message['status']
            del message['x-status']
        elif isinstance(message, BabylMessage):
            flags = set(self.get_flags())
            if 'R' not in flags:
                message.add_label('unseen')
            if 'D' in flags:
                message.add_label('deleted')
            if 'A' in flags:
                message.add_label('answered')
            del message['status']
            del message['x-status']
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' % type(message))