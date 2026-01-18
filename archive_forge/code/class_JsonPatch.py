from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
class JsonPatch(object):
    json_dumper = staticmethod(json.dumps)
    json_loader = staticmethod(_jsonloads)
    operations = MappingProxyType({'remove': RemoveOperation, 'add': AddOperation, 'replace': ReplaceOperation, 'move': MoveOperation, 'test': TestOperation, 'copy': CopyOperation})
    "A JSON Patch is a list of Patch Operations.\n\n    >>> patch = JsonPatch([\n    ...     {'op': 'add', 'path': '/foo', 'value': 'bar'},\n    ...     {'op': 'add', 'path': '/baz', 'value': [1, 2, 3]},\n    ...     {'op': 'remove', 'path': '/baz/1'},\n    ...     {'op': 'test', 'path': '/baz', 'value': [1, 3]},\n    ...     {'op': 'replace', 'path': '/baz/0', 'value': 42},\n    ...     {'op': 'remove', 'path': '/baz/1'},\n    ... ])\n    >>> doc = {}\n    >>> result = patch.apply(doc)\n    >>> expected = {'foo': 'bar', 'baz': [42]}\n    >>> result == expected\n    True\n\n    JsonPatch object is iterable, so you can easily access each patch\n    statement in a loop:\n\n    >>> lpatch = list(patch)\n    >>> expected = {'op': 'add', 'path': '/foo', 'value': 'bar'}\n    >>> lpatch[0] == expected\n    True\n    >>> lpatch == patch.patch\n    True\n\n    Also JsonPatch could be converted directly to :class:`bool` if it contains\n    any operation statements:\n\n    >>> bool(patch)\n    True\n    >>> bool(JsonPatch([]))\n    False\n\n    This behavior is very handy with :func:`make_patch` to write more readable\n    code:\n\n    >>> old = {'foo': 'bar', 'numbers': [1, 3, 4, 8]}\n    >>> new = {'baz': 'qux', 'numbers': [1, 4, 7]}\n    >>> patch = make_patch(old, new)\n    >>> if patch:\n    ...     # document have changed, do something useful\n    ...     patch.apply(old)    #doctest: +ELLIPSIS\n    {...}\n    "

    def __init__(self, patch, pointer_cls=JsonPointer):
        self.patch = patch
        self.pointer_cls = pointer_cls
        for op in self.patch:
            if isinstance(op, basestring):
                raise InvalidJsonPatch('Document is expected to be sequence of operations, got a sequence of strings.')
            self._get_operation(op)

    def __str__(self):
        """str(self) -> self.to_string()"""
        return self.to_string()

    def __bool__(self):
        return bool(self.patch)
    __nonzero__ = __bool__

    def __iter__(self):
        return iter(self.patch)

    def __hash__(self):
        return hash(tuple(self._ops))

    def __eq__(self, other):
        if not isinstance(other, JsonPatch):
            return False
        return self._ops == other._ops

    def __ne__(self, other):
        return not self == other

    @classmethod
    def from_string(cls, patch_str, loads=None, pointer_cls=JsonPointer):
        """Creates JsonPatch instance from string source.

        :param patch_str: JSON patch as raw string.
        :type patch_str: str

        :param loads: A function of one argument that loads a serialized
                      JSON string.
        :type loads: function

        :param pointer_cls: JSON pointer class to use.
        :type pointer_cls: Type[JsonPointer]

        :return: :class:`JsonPatch` instance.
        """
        json_loader = loads or cls.json_loader
        patch = json_loader(patch_str)
        return cls(patch, pointer_cls=pointer_cls)

    @classmethod
    def from_diff(cls, src, dst, optimization=True, dumps=None, pointer_cls=JsonPointer):
        """Creates JsonPatch instance based on comparison of two document
        objects. Json patch would be created for `src` argument against `dst`
        one.

        :param src: Data source document object.
        :type src: dict

        :param dst: Data source document object.
        :type dst: dict

        :param dumps: A function of one argument that produces a serialized
                      JSON string.
        :type dumps: function

        :param pointer_cls: JSON pointer class to use.
        :type pointer_cls: Type[JsonPointer]

        :return: :class:`JsonPatch` instance.

        >>> src = {'foo': 'bar', 'numbers': [1, 3, 4, 8]}
        >>> dst = {'baz': 'qux', 'numbers': [1, 4, 7]}
        >>> patch = JsonPatch.from_diff(src, dst)
        >>> new = patch.apply(src)
        >>> new == dst
        True
        """
        json_dumper = dumps or cls.json_dumper
        builder = DiffBuilder(src, dst, json_dumper, pointer_cls=pointer_cls)
        builder._compare_values('', None, src, dst)
        ops = list(builder.execute())
        return cls(ops, pointer_cls=pointer_cls)

    def to_string(self, dumps=None):
        """Returns patch set as JSON string."""
        json_dumper = dumps or self.json_dumper
        return json_dumper(self.patch)

    @property
    def _ops(self):
        return tuple(map(self._get_operation, self.patch))

    def apply(self, obj, in_place=False):
        """Applies the patch to a given object.

        :param obj: Document object.
        :type obj: dict

        :param in_place: Tweaks the way how patch would be applied - directly to
                         specified `obj` or to its copy.
        :type in_place: bool

        :return: Modified `obj`.
        """
        if not in_place:
            obj = copy.deepcopy(obj)
        for operation in self._ops:
            obj = operation.apply(obj)
        return obj

    def _get_operation(self, operation):
        if 'op' not in operation:
            raise InvalidJsonPatch("Operation does not contain 'op' member")
        op = operation['op']
        if not isinstance(op, basestring):
            raise InvalidJsonPatch("Operation's op must be a string")
        if op not in self.operations:
            raise InvalidJsonPatch('Unknown operation {0!r}'.format(op))
        cls = self.operations[op]
        return cls(operation, pointer_cls=self.pointer_cls)