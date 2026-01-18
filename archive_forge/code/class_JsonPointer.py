from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
class JsonPointer(object):
    """A JSON Pointer that can reference parts of a JSON document"""
    _RE_ARRAY_INDEX = re.compile('0|[1-9][0-9]*$')
    _RE_INVALID_ESCAPE = re.compile('(~[^01]|~$)')

    def __init__(self, pointer):
        invalid_escape = self._RE_INVALID_ESCAPE.search(pointer)
        if invalid_escape:
            raise JsonPointerException('Found invalid escape {}'.format(invalid_escape.group()))
        parts = pointer.split('/')
        if parts.pop(0) != '':
            raise JsonPointerException('Location must start with /')
        parts = [unescape(part) for part in parts]
        self.parts = parts

    def to_last(self, doc):
        """Resolves ptr until the last step, returns (sub-doc, last-step)"""
        if not self.parts:
            return (doc, None)
        for part in self.parts[:-1]:
            doc = self.walk(doc, part)
        return (doc, JsonPointer.get_part(doc, self.parts[-1]))

    def resolve(self, doc, default=_nothing):
        """Resolves the pointer against doc and returns the referenced object"""
        for part in self.parts:
            try:
                doc = self.walk(doc, part)
            except JsonPointerException:
                if default is _nothing:
                    raise
                else:
                    return default
        return doc
    get = resolve

    def set(self, doc, value, inplace=True):
        """Resolve the pointer against the doc and replace the target with value."""
        if len(self.parts) == 0:
            if inplace:
                raise JsonPointerException('Cannot set root in place')
            return value
        if not inplace:
            doc = copy.deepcopy(doc)
        parent, part = self.to_last(doc)
        if isinstance(parent, Sequence) and part == '-':
            parent.append(value)
        else:
            parent[part] = value
        return doc

    @classmethod
    def get_part(cls, doc, part):
        """Returns the next step in the correct type"""
        if isinstance(doc, Mapping):
            return part
        elif isinstance(doc, Sequence):
            if part == '-':
                return part
            if not JsonPointer._RE_ARRAY_INDEX.match(str(part)):
                raise JsonPointerException("'%s' is not a valid sequence index" % part)
            return int(part)
        elif hasattr(doc, '__getitem__'):
            return part
        else:
            raise JsonPointerException("Document '%s' does not support indexing, must be mapping/sequence or support __getitem__" % type(doc))

    def get_parts(self):
        """Returns the list of the parts. For example, JsonPointer('/a/b').get_parts() == ['a', 'b']"""
        return self.parts

    def walk(self, doc, part):
        """ Walks one step in doc and returns the referenced part """
        part = JsonPointer.get_part(doc, part)
        assert hasattr(doc, '__getitem__'), 'invalid document type %s' % (type(doc),)
        if isinstance(doc, Sequence):
            if part == '-':
                return EndOfList(doc)
            try:
                return doc[part]
            except IndexError:
                raise JsonPointerException("index '%s' is out of bounds" % (part,))
        try:
            return doc[part]
        except KeyError:
            raise JsonPointerException("member '%s' not found in %s" % (part, doc))

    def contains(self, ptr):
        """ Returns True if self contains the given ptr """
        return self.parts[:len(ptr.parts)] == ptr.parts

    def __contains__(self, item):
        """ Returns True if self contains the given ptr """
        return self.contains(item)

    def join(self, suffix):
        """ Returns a new JsonPointer with the given suffix append to this ptr """
        if isinstance(suffix, JsonPointer):
            suffix_parts = suffix.parts
        elif isinstance(suffix, str):
            suffix_parts = JsonPointer(suffix).parts
        else:
            suffix_parts = suffix
        try:
            return JsonPointer.from_parts(chain(self.parts, suffix_parts))
        except:
            raise JsonPointerException('Invalid suffix')

    def __truediv__(self, suffix):
        return self.join(suffix)
    __div__ = __truediv__

    @property
    def path(self):
        """Returns the string representation of the pointer

        >>> ptr = JsonPointer('/~0/0/~1').path == '/~0/0/~1'
        """
        parts = [escape(part) for part in self.parts]
        return ''.join(('/' + part for part in parts))

    def __eq__(self, other):
        """Compares a pointer to another object

        Pointers can be compared by comparing their strings (or splitted
        strings), because no two different parts can point to the same
        structure in an object (eg no different number representations)
        """
        if not isinstance(other, JsonPointer):
            return False
        return self.parts == other.parts

    def __hash__(self):
        return hash(tuple(self.parts))

    def __str__(self):
        return encode_str(self.path)

    def __repr__(self):
        return 'JsonPointer(' + repr(self.path) + ')'

    @classmethod
    def from_parts(cls, parts):
        """Constructs a JsonPointer from a list of (unescaped) paths

        >>> JsonPointer.from_parts(['a', '~', '/', 0]).path == '/a/~0/~1/0'
        True
        """
        parts = [escape(str(part)) for part in parts]
        ptr = cls(''.join(('/' + part for part in parts)))
        return ptr