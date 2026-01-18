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
class BabylMessage(Message):
    """Message with Babyl-specific properties."""
    _type_specific_attributes = ['_labels', '_visible']

    def __init__(self, message=None):
        """Initialize a BabylMessage instance."""
        self._labels = []
        self._visible = Message()
        Message.__init__(self, message)

    def get_labels(self):
        """Return a list of labels on the message."""
        return self._labels[:]

    def set_labels(self, labels):
        """Set the list of labels on the message."""
        self._labels = list(labels)

    def add_label(self, label):
        """Add label to list of labels on the message."""
        if isinstance(label, str):
            if label not in self._labels:
                self._labels.append(label)
        else:
            raise TypeError('label must be a string: %s' % type(label))

    def remove_label(self, label):
        """Remove label from the list of labels on the message."""
        try:
            self._labels.remove(label)
        except ValueError:
            pass

    def get_visible(self):
        """Return a Message representation of visible headers."""
        return Message(self._visible)

    def set_visible(self, visible):
        """Set the Message representation of visible headers."""
        self._visible = Message(visible)

    def update_visible(self):
        """Update and/or sensibly generate a set of visible headers."""
        for header in self._visible.keys():
            if header in self:
                self._visible.replace_header(header, self[header])
            else:
                del self._visible[header]
        for header in ('Date', 'From', 'Reply-To', 'To', 'CC', 'Subject'):
            if header in self and header not in self._visible:
                self._visible[header] = self[header]

    def _explain_to(self, message):
        """Copy Babyl-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            labels = set(self.get_labels())
            if 'unseen' in labels:
                message.set_subdir('cur')
            else:
                message.set_subdir('cur')
                message.add_flag('S')
            if 'forwarded' in labels or 'resent' in labels:
                message.add_flag('P')
            if 'answered' in labels:
                message.add_flag('R')
            if 'deleted' in labels:
                message.add_flag('T')
        elif isinstance(message, _mboxMMDFMessage):
            labels = set(self.get_labels())
            if 'unseen' not in labels:
                message.add_flag('RO')
            else:
                message.add_flag('O')
            if 'deleted' in labels:
                message.add_flag('D')
            if 'answered' in labels:
                message.add_flag('A')
        elif isinstance(message, MHMessage):
            labels = set(self.get_labels())
            if 'unseen' in labels:
                message.add_sequence('unseen')
            if 'answered' in labels:
                message.add_sequence('replied')
        elif isinstance(message, BabylMessage):
            message.set_visible(self.get_visible())
            for label in self.get_labels():
                message.add_label(label)
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' % type(message))