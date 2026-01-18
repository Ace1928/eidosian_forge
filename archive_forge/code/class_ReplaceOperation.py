from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class ReplaceOperation(PatchOperation):
    """Replaces an object property or an array element by a new value."""

    def apply(self, obj):
        try:
            value = self.operation['value']
        except KeyError as ex:
            raise InvalidJsonPatch("The operation does not contain a 'value' member")
        subobj, part = self.pointer.to_last(obj)
        if part is None:
            return value
        if part == '-':
            raise InvalidJsonPatch("'path' with '-' can't be applied to 'replace' operation")
        if isinstance(subobj, MutableSequence):
            if part >= len(subobj) or part < 0:
                raise JsonPatchConflict("can't replace outside of list")
        elif isinstance(subobj, MutableMapping):
            if part not in subobj:
                msg = "can't replace a non-existent object '{0}'".format(part)
                raise JsonPatchConflict(msg)
        elif part is None:
            raise TypeError('invalid document type {0}'.format(type(subobj)))
        else:
            raise JsonPatchConflict('unable to fully resolve json pointer {0}, part {1}'.format(self.location, part))
        subobj[part] = value
        return obj

    def _on_undo_remove(self, path, key):
        return key

    def _on_undo_add(self, path, key):
        return key