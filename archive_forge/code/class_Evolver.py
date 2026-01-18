from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
class Evolver(object):
    __slots__ = ('_count', '_shift', '_root', '_tail', '_tail_offset', '_dirty_nodes', '_extra_tail', '_cached_leafs', '_orig_pvector')

    def __init__(self, v):
        self._reset(v)

    def __getitem__(self, index):
        if not isinstance(index, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
        if index < 0:
            index += self._count + len(self._extra_tail)
        if self._count <= index < self._count + len(self._extra_tail):
            return self._extra_tail[index - self._count]
        return PythonPVector._node_for(self, index)[index & BIT_MASK]

    def _reset(self, v):
        self._count = v._count
        self._shift = v._shift
        self._root = v._root
        self._tail = v._tail
        self._tail_offset = v._tail_offset
        self._dirty_nodes = {}
        self._cached_leafs = {}
        self._extra_tail = []
        self._orig_pvector = v

    def append(self, element):
        self._extra_tail.append(element)
        return self

    def extend(self, iterable):
        self._extra_tail.extend(iterable)
        return self

    def set(self, index, val):
        self[index] = val
        return self

    def __setitem__(self, index, val):
        if not isinstance(index, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)
        if index < 0:
            index += self._count + len(self._extra_tail)
        if 0 <= index < self._count:
            node = self._cached_leafs.get(index >> SHIFT)
            if node:
                node[index & BIT_MASK] = val
            elif index >= self._tail_offset:
                if id(self._tail) not in self._dirty_nodes:
                    self._tail = list(self._tail)
                    self._dirty_nodes[id(self._tail)] = True
                    self._cached_leafs[index >> SHIFT] = self._tail
                self._tail[index & BIT_MASK] = val
            else:
                self._root = self._do_set(self._shift, self._root, index, val)
        elif self._count <= index < self._count + len(self._extra_tail):
            self._extra_tail[index - self._count] = val
        elif index == self._count + len(self._extra_tail):
            self._extra_tail.append(val)
        else:
            raise IndexError('Index out of range: %s' % (index,))

    def _do_set(self, level, node, i, val):
        if id(node) in self._dirty_nodes:
            ret = node
        else:
            ret = list(node)
            self._dirty_nodes[id(ret)] = True
        if level == 0:
            ret[i & BIT_MASK] = val
            self._cached_leafs[i >> SHIFT] = ret
        else:
            sub_index = i >> level & BIT_MASK
            ret[sub_index] = self._do_set(level - SHIFT, node[sub_index], i, val)
        return ret

    def delete(self, index):
        del self[index]
        return self

    def __delitem__(self, key):
        if self._orig_pvector:
            l = PythonPVector(self._count, self._shift, self._root, self._tail).tolist()
            l.extend(self._extra_tail)
            self._reset(_EMPTY_PVECTOR)
            self._extra_tail = l
        del self._extra_tail[key]

    def persistent(self):
        result = self._orig_pvector
        if self.is_dirty():
            result = PythonPVector(self._count, self._shift, self._root, self._tail).extend(self._extra_tail)
            self._reset(result)
        return result

    def __len__(self):
        return self._count + len(self._extra_tail)

    def is_dirty(self):
        return bool(self._dirty_nodes or self._extra_tail)