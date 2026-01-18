import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class Conflict(BaseConflict):
    """Base class for all types of conflict"""
    has_files = False

    def __init__(self, path, file_id=None):
        super().__init__(path)
        if isinstance(file_id, str):
            file_id = cache_utf8.encode(file_id)
        self.file_id = file_id

    def as_stanza(self):
        s = rio.Stanza(type=self.typestring, path=self.path)
        if self.file_id is not None:
            s.add('file_id', self.file_id.decode('utf8'))
        return s

    def _cmp_list(self):
        return [self.typestring, self.path, self.file_id]

    def __eq__(self, other):
        if getattr(other, '_cmp_list', None) is None:
            return False
        x = self._cmp_list()
        y = other._cmp_list()
        return x == y

    def __hash__(self):
        return hash((type(self), self.path, self.file_id))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __unicode__(self):
        return self.describe()

    def __str__(self):
        return self.describe()

    def describe(self):
        return self.format % self.__dict__

    def __repr__(self):
        rdict = dict(self.__dict__)
        rdict['class'] = self.__class__.__name__
        return self.rformat % rdict

    @staticmethod
    def factory(type, **kwargs):
        global ctype
        return ctype[type](**kwargs)

    @staticmethod
    def sort_key(conflict):
        if conflict.path is not None:
            return (conflict.path, conflict.typestring)
        elif getattr(conflict, 'conflict_path', None) is not None:
            return (conflict.conflict_path, conflict.typestring)
        else:
            return (None, conflict.typestring)

    def do(self, action, tree):
        """Apply the specified action to the conflict.

        :param action: The method name to call.

        :param tree: The tree passed as a parameter to the method.
        """
        meth = getattr(self, 'action_%s' % action, None)
        if meth is None:
            raise NotImplementedError(self.__class__.__name__ + '.' + action)
        meth(tree)

    def action_auto(self, tree):
        raise NotImplementedError(self.action_auto)

    def action_done(self, tree):
        """Mark the conflict as solved once it has been handled."""
        pass

    def action_take_this(self, tree):
        raise NotImplementedError(self.action_take_this)

    def action_take_other(self, tree):
        raise NotImplementedError(self.action_take_other)

    def _resolve_with_cleanups(self, tree, *args, **kwargs):
        with tree.transform() as tt:
            self._resolve(tt, *args, **kwargs)