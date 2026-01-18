import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _get_set_or_delete(self, id, operation, new_item=None):
    _check_if_has_indexable_children(self)
    if isinstance(self.children, Component):
        if getattr(self.children, 'id', None) is not None:
            if self.children.id == id:
                if operation == 'get':
                    return self.children
                if operation == 'set':
                    self.children = new_item
                    return
                if operation == 'delete':
                    self.children = None
                    return
        try:
            if operation == 'get':
                return self.children.__getitem__(id)
            if operation == 'set':
                self.children.__setitem__(id, new_item)
                return
            if operation == 'delete':
                self.children.__delitem__(id)
                return
        except KeyError:
            pass
    if isinstance(self.children, (tuple, MutableSequence)):
        for i, item in enumerate(self.children):
            if getattr(item, 'id', None) == id:
                if operation == 'get':
                    return item
                if operation == 'set':
                    self.children[i] = new_item
                    return
                if operation == 'delete':
                    del self.children[i]
                    return
            elif isinstance(item, Component):
                try:
                    if operation == 'get':
                        return item.__getitem__(id)
                    if operation == 'set':
                        item.__setitem__(id, new_item)
                        return
                    if operation == 'delete':
                        item.__delitem__(id)
                        return
                except KeyError:
                    pass
    raise KeyError(id)