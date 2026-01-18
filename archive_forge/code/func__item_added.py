from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def _item_added(self, path, key, item):
    index = self.take_index(item, _ST_REMOVE)
    if index is not None:
        op = index[2]
        if type(op.key) == int and type(key) == int:
            for v in self.iter_from(index):
                op.key = v._on_undo_remove(op.path, op.key)
        self.remove(index)
        if op.location != _path_join(path, key):
            new_op = MoveOperation({'op': 'move', 'from': op.location, 'path': _path_join(path, key)}, pointer_cls=self.pointer_cls)
            self.insert(new_op)
    else:
        new_op = AddOperation({'op': 'add', 'path': _path_join(path, key), 'value': item}, pointer_cls=self.pointer_cls)
        new_index = self.insert(new_op)
        self.store_index(item, new_index, _ST_ADD)