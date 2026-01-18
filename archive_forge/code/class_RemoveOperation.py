from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class RemoveOperation(PatchOperation):
    """Removes an object property or an array element."""

    def apply(self, obj):
        subobj, part = self.pointer.to_last(obj)
        if isinstance(subobj, Sequence) and (not isinstance(part, int)):
            raise JsonPointerException("invalid array index '{0}'".format(part))
        try:
            del subobj[part]
        except (KeyError, IndexError) as ex:
            msg = "can't remove a non-existent object '{0}'".format(part)
            raise JsonPatchConflict(msg)
        return obj

    def _on_undo_remove(self, path, key):
        if self.path == path:
            if self.key >= key:
                self.key += 1
            else:
                key -= 1
        return key

    def _on_undo_add(self, path, key):
        if self.path == path:
            if self.key > key:
                self.key -= 1
            else:
                key -= 1
        return key