from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class TestOperation(PatchOperation):
    """Test value by specified location."""

    def apply(self, obj):
        try:
            subobj, part = self.pointer.to_last(obj)
            if part is None:
                val = subobj
            else:
                val = self.pointer.walk(subobj, part)
        except JsonPointerException as ex:
            raise JsonPatchTestFailed(str(ex))
        try:
            value = self.operation['value']
        except KeyError as ex:
            raise InvalidJsonPatch("The operation does not contain a 'value' member")
        if val != value:
            msg = '{0} ({1}) is not equal to tested value {2} ({3})'
            raise JsonPatchTestFailed(msg.format(val, type(val), value, type(value)))
        return obj