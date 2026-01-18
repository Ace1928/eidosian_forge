from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class AddOperation(PatchOperation):
    """Adds an object property or an array element."""

    def apply(self, obj):
        try:
            value = self.operation['value']
        except KeyError as ex:
            raise InvalidJsonPatch("The operation does not contain a 'value' member")
        subobj, part = self.pointer.to_last(obj)
        if isinstance(subobj, MutableSequence):
            if part == '-':
                subobj.append(value)
            elif part > len(subobj) or part < 0:
                raise JsonPatchConflict("can't insert outside of list")
            else:
                subobj.insert(part, value)
        elif isinstance(subobj, MutableMapping):
            if part is None:
                obj = value
            else:
                subobj[part] = value
        elif part is None:
            raise TypeError('invalid document type {0}'.format(type(subobj)))
        else:
            raise JsonPatchConflict('unable to fully resolve json pointer {0}, part {1}'.format(self.location, part))
        return obj

    def _on_undo_remove(self, path, key):
        if self.path == path:
            if self.key > key:
                self.key += 1
            else:
                key += 1
        return key

    def _on_undo_add(self, path, key):
        if self.path == path:
            if self.key > key:
                self.key -= 1
            else:
                key += 1
        return key