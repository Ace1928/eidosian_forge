from __future__ import division
import re
import stat
from .helpers import (
class ImportCommand(object):
    """Base class for import commands."""

    def __init__(self, name):
        self.name = name
        self._binary = []

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return bytes(self).decode('utf8')

    def __bytes__(self):
        raise NotImplementedError('An implementation of __bytes__ is required')

    def dump_str(self, names=None, child_lists=None, verbose=False):
        """Dump fields as a string.

        For debugging.

        :param names: the list of fields to include or
            None for all public fields
        :param child_lists: dictionary of child command names to
            fields for that child command to include
        :param verbose: if True, prefix each line with the command class and
            display fields as a dictionary; if False, dump just the field
            values with tabs between them
        """
        interesting = {}
        if names is None:
            fields = [k for k in list(self.__dict__.keys()) if not k.startswith(b'_')]
        else:
            fields = names
        for field in fields:
            value = self.__dict__.get(field)
            if field in self._binary and value is not None:
                value = b'(...)'
            interesting[field] = value
        if verbose:
            return '%s: %s' % (self.__class__.__name__, interesting)
        else:
            return '\t'.join([repr(interesting[k]) for k in fields])