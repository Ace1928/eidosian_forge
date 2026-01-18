from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class DiffBuilder(object):

    def __init__(self, src_doc, dst_doc, dumps=json.dumps, pointer_cls=JsonPointer):
        self.dumps = dumps
        self.pointer_cls = pointer_cls
        self.index_storage = [{}, {}]
        self.index_storage2 = [[], []]
        self.__root = root = []
        self.src_doc = src_doc
        self.dst_doc = dst_doc
        root[:] = [root, root, None]

    def store_index(self, value, index, st):
        typed_key = (value, type(value))
        try:
            storage = self.index_storage[st]
            stored = storage.get(typed_key)
            if stored is None:
                storage[typed_key] = [index]
            else:
                storage[typed_key].append(index)
        except TypeError:
            self.index_storage2[st].append((typed_key, index))

    def take_index(self, value, st):
        typed_key = (value, type(value))
        try:
            stored = self.index_storage[st].get(typed_key)
            if stored:
                return stored.pop()
        except TypeError:
            storage = self.index_storage2[st]
            for i in range(len(storage) - 1, -1, -1):
                if storage[i][0] == typed_key:
                    return storage.pop(i)[1]

    def insert(self, op):
        root = self.__root
        last = root[0]
        last[1] = root[0] = [last, root, op]
        return root[0]

    def remove(self, index):
        link_prev, link_next, _ = index
        link_prev[1] = link_next
        link_next[0] = link_prev
        index[:] = []

    def iter_from(self, start):
        root = self.__root
        curr = start[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]

    def __iter__(self):
        root = self.__root
        curr = root[1]
        while curr is not root:
            yield curr[2]
            curr = curr[1]

    def execute(self):
        root = self.__root
        curr = root[1]
        while curr is not root:
            if curr[1] is not root:
                op_first, op_second = (curr[2], curr[1][2])
                if op_first.location == op_second.location and type(op_first) == RemoveOperation and (type(op_second) == AddOperation):
                    yield ReplaceOperation({'op': 'replace', 'path': op_second.location, 'value': op_second.operation['value']}, pointer_cls=self.pointer_cls).operation
                    curr = curr[1][1]
                    continue
            yield curr[2].operation
            curr = curr[1]

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

    def _item_removed(self, path, key, item):
        new_op = RemoveOperation({'op': 'remove', 'path': _path_join(path, key)}, pointer_cls=self.pointer_cls)
        index = self.take_index(item, _ST_ADD)
        new_index = self.insert(new_op)
        if index is not None:
            op = index[2]
            added_item = op.pointer.to_last(self.dst_doc)[0]
            if type(added_item) == list:
                for v in self.iter_from(index):
                    op.key = v._on_undo_add(op.path, op.key)
            self.remove(index)
            if new_op.location != op.location:
                new_op = MoveOperation({'op': 'move', 'from': new_op.location, 'path': op.location}, pointer_cls=self.pointer_cls)
                new_index[2] = new_op
            else:
                self.remove(new_index)
        else:
            self.store_index(item, new_index, _ST_REMOVE)

    def _item_replaced(self, path, key, item):
        self.insert(ReplaceOperation({'op': 'replace', 'path': _path_join(path, key), 'value': item}, pointer_cls=self.pointer_cls))

    def _compare_dicts(self, path, src, dst):
        src_keys = set(src.keys())
        dst_keys = set(dst.keys())
        added_keys = dst_keys - src_keys
        removed_keys = src_keys - dst_keys
        for key in removed_keys:
            self._item_removed(path, str(key), src[key])
        for key in added_keys:
            self._item_added(path, str(key), dst[key])
        for key in src_keys & dst_keys:
            self._compare_values(path, key, src[key], dst[key])

    def _compare_lists(self, path, src, dst):
        len_src, len_dst = (len(src), len(dst))
        max_len = max(len_src, len_dst)
        min_len = min(len_src, len_dst)
        for key in range(max_len):
            if key < min_len:
                old, new = (src[key], dst[key])
                if old == new:
                    continue
                elif isinstance(old, MutableMapping) and isinstance(new, MutableMapping):
                    self._compare_dicts(_path_join(path, key), old, new)
                elif isinstance(old, MutableSequence) and isinstance(new, MutableSequence):
                    self._compare_lists(_path_join(path, key), old, new)
                else:
                    self._item_removed(path, key, old)
                    self._item_added(path, key, new)
            elif len_src > len_dst:
                self._item_removed(path, len_dst, src[key])
            else:
                self._item_added(path, key, dst[key])

    def _compare_values(self, path, key, src, dst):
        if isinstance(src, MutableMapping) and isinstance(dst, MutableMapping):
            self._compare_dicts(_path_join(path, key), src, dst)
        elif isinstance(src, MutableSequence) and isinstance(dst, MutableSequence):
            self._compare_lists(_path_join(path, key), src, dst)
        elif self.dumps(src) == self.dumps(dst):
            return
        else:
            self._item_replaced(path, key, dst)