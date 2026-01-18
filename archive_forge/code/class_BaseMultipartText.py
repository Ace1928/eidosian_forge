from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
@py3compat.python_2_unicode_compatible
class BaseMultipartText(BaseText):
    info = ()

    def __init__(self, *parts):
        """Create a text object consisting of one or more parts.

        Empty parts are ignored:

        >>> Text() == Text('') == Text('', '', '')
        True
        >>> Text('Word', '') == Text('Word')
        True

        Text() objects are unpacked and their children are included directly:

        >>> Text(Text('Multi', ' '), Tag('em', 'part'), Text(' ', Text('text!')))
        Text('Multi ', Tag('em', 'part'), ' text!')
        >>> Tag('strong', Text('Multi', ' '), Tag('em', 'part'), Text(' ', 'text!'))
        Tag('strong', 'Multi ', Tag('em', 'part'), ' text!')

        Similar objects are merged together:

        >>> Text('Multi', Tag('em', 'part'), Text(Tag('em', ' ', 'text!')))
        Text('Multi', Tag('em', 'part text!'))
        >>> Text('Please ', HRef('/', 'click'), HRef('/', ' here'), '.')
        Text('Please ', HRef('/', 'click here'), '.')
        """
        parts = (ensure_text(part) for part in parts)
        nonempty_parts = (part for part in parts if part)
        unpacked_parts = itertools.chain(*[part._unpack() for part in nonempty_parts])
        merged_parts = self._merge_similar(unpacked_parts)
        self.parts = list(merged_parts)
        self.length = sum((len(part) for part in self.parts))

    def __str__(self):
        return ''.join((six.text_type(part) for part in self.parts))

    def __eq__(self, other):
        """
        Rich text objects support equality comparison:

        >>> Text('Cat') == Text('cat')
        False
        >>> Text('Cat') == Text('Cat')
        True

        """
        return isinstance(other, BaseText) and self._typeinfo() == other._typeinfo() and (self.parts == other.parts)

    def __len__(self):
        """
        ``len(text)`` returns the number of characters in the text, ignoring
        the markup:

        >>> len(Text('Long cat'))
        8
        >>> len(Text(Tag('em', 'Long'), ' cat'))
        8
        >>> len(Text(HRef('http://example.com/', 'Long'), ' cat'))
        8

        """
        return self.length

    def __contains__(self, item):
        """
        ``value in text`` returns ``True`` if any part of the ``text``
        contains the substring ``value``:

        >>> 'Long cat' in Text('Long cat!')
        True

        Substrings splitted across multiple text parts are not matched:

        >>> 'Long cat' in Text(Tag('em', 'Long'), 'cat!')
        False

        """
        if not isinstance(item, six.string_types):
            raise TypeError(item)
        return not item or any((part.__contains__(item) for part in self.parts))

    def __getitem__(self, key):
        """
        Slicing and extracting characters works like with regular strings,
        formatting is preserved.

        >>> Text('Longcat is ', Tag('em', 'looooooong!'))[:15]
        Text('Longcat is ', Tag('em', 'looo'))
        >>> Text('Longcat is ', Tag('em', 'looooooong!'))[-1]
        Text(Tag('em', '!'))
        """
        if isinstance(key, six.integer_types):
            start = key
            end = None
        elif isinstance(key, slice):
            start, end, step = key.indices(len(self))
            if step != 1:
                raise NotImplementedError
        else:
            raise TypeError(key, type(key))
        if start < 0:
            start = len(self) + start
        if end is None:
            end = start + 1
        if end < 0:
            end = len(self) + end
        return self._slice_end(len(self) - start)._slice_beginning(end - start)

    def _slice_beginning(self, slice_length):
        """
        Return a text consistng of the first slice_length characters
        of this text (with formatting preserved).
        """
        parts = []
        length = 0
        for part in self.parts:
            if length + len(part) > slice_length:
                parts.append(part[:slice_length - length])
                break
            else:
                parts.append(part)
                length += len(part)
        return self._create_similar(parts)

    def _slice_end(self, slice_length):
        """
        Return a text consistng of the last slice_length characters
        of this text (with formatting preserved).
        """
        parts = []
        length = 0
        for part in reversed(self.parts):
            if length + len(part) > slice_length:
                parts.append(part[len(part) - (slice_length - length):])
                break
            else:
                parts.append(part)
                length += len(part)
        return self._create_similar(reversed(parts))

    def append(self, text):
        """
        Append text to the end of this text.

        For Tags, HRefs, etc. the appended text is placed *inside* the tag.

        >>> text = Tag('strong', 'Chuck Norris')
        >>> print((text +  ' wins!').render_as('html'))
        <strong>Chuck Norris</strong> wins!
        >>> print(text.append(' wins!').render_as('html'))
        <strong>Chuck Norris wins!</strong>
        """
        return self._create_similar(self.parts + [text])

    @collect_iterable
    def split(self, sep=None, keep_empty_parts=None):
        """
        >>> Text('a + b').split()
        [Text('a'), Text('+'), Text('b')]

        >>> Text('a, b').split(', ')
        [Text('a'), Text('b')]
        """
        if keep_empty_parts is None:
            keep_empty_parts = sep is not None
        tail = [''] if keep_empty_parts else []
        for part in self.parts:
            split_part = part.split(sep, keep_empty_parts=True)
            if not split_part:
                continue
            for item in split_part[:-1]:
                if tail:
                    yield self._create_similar(tail + [item])
                    tail = []
                elif item or keep_empty_parts:
                    yield self._create_similar([item])
            tail.append(split_part[-1])
        if tail:
            tail_text = self._create_similar(tail)
            if tail_text or keep_empty_parts:
                yield tail_text

    def startswith(self, prefix):
        """
        Return True if the text starts with the given prefix.

        >>> Text('Longcat!').startswith('Longcat')
        True

        Prefixes split across multiple parts are not matched:

        >>> Text(Tag('em', 'Long'), 'cat!').startswith('Longcat')
        False

        """
        if not self.parts:
            return False
        else:
            return self.parts[0].startswith(prefix)

    def endswith(self, suffix):
        """
        Return True if the text ends with the given suffix.

        >>> Text('Longcat!').endswith('cat!')
        True

        Suffixes split across multiple parts are not matched:

        >>> Text('Long', Tag('em', 'cat'), '!').endswith('cat!')
        False

        """
        if not self.parts:
            return False
        else:
            return self.parts[-1].endswith(suffix)

    def isalpha(self):
        """
        Return True if all characters in the string are alphabetic and there is
        at least one character, False otherwise.
        """
        return bool(self) and all((part.isalpha() for part in self.parts))

    def lower(self):
        """
        Convert rich text to lowercase.

        >>> Text(Tag('em', 'Long cat')).lower()
        Text(Tag('em', 'long cat'))
        """
        return self._create_similar((part.lower() for part in self.parts))

    def upper(self):
        """
        Convert rich text to uppsercase.

        >>> Text(Tag('em', 'Long cat')).upper()
        Text(Tag('em', 'LONG CAT'))
        """
        return self._create_similar((part.upper() for part in self.parts))

    def render(self, backend):
        """
        Render this :py:class:`Text` into markup.

        :param backend: The formatting backend (an instance of
            :py:class:`pybtex.backends.BaseBackend`).
        """
        rendered_list = [part.render(backend) for part in self.parts]
        assert all((isinstance(item, backend.RenderType) for item in rendered_list))
        return backend.render_sequence(rendered_list)

    def _typeinfo(self):
        """Return the type and the parameters used to create this text object.

        >>> text = Tag('strong', 'Heavy rain!')
        >>> text._typeinfo() == (Tag, ('strong',))
        True

        """
        return (type(self), self.info)

    def _create_similar(self, parts):
        """
        Create a new text object of the same type with the same parameters,
        with different text content.

        >>> text = Tag('strong', 'Bananas!')
        >>> text._create_similar(['Apples!'])
        Tag('strong', 'Apples!')
        """
        cls, cls_args = self._typeinfo()
        args = list(cls_args) + list(parts)
        return cls(*args)

    def _merge_similar(self, parts):
        """Merge adjacent text objects with the same type and parameters together.

        >>> text = Text()
        >>> parts = [Tag('em', 'Breaking'), Tag('em', ' '), Tag('em', 'news!')]
        >>> list(text._merge_similar(parts))
        [Tag('em', 'Breaking news!')]
        """
        groups = itertools.groupby(parts, lambda value: value._typeinfo())
        for typeinfo, group in groups:
            cls, info = typeinfo
            group = list(group)
            if cls and len(group) > 1:
                group_parts = itertools.chain(*(text.parts for text in group))
                args = list(info) + list(group_parts)
                yield cls(*args)
            else:
                for text in group:
                    yield text

    @deprecated('0.19', 'use __unicode__() instead')
    def plaintext(self):
        return six.text_type(self)

    @deprecated('0.19')
    def enumerate(self):
        for n, child in enumerate(self.parts):
            try:
                for p in child.enumerate():
                    yield p
            except AttributeError:
                yield (self, n)

    @deprecated('0.19')
    def reversed(self):
        for n, child in reversed(list(enumerate(self.parts))):
            try:
                for p in child.reversed():
                    yield p
            except AttributeError:
                yield (self, n)

    @deprecated('0.19', 'use slicing instead')
    def get_beginning(self):
        try:
            l, i = next(self.enumerate())
        except StopIteration:
            pass
        else:
            return l.parts[i]

    @deprecated('0.19', 'use slicing instead')
    def get_end(self):
        try:
            l, i = next(self.reversed())
        except StopIteration:
            pass
        else:
            return l.parts[i]

    @deprecated('0.19', 'use slicing instead')
    def apply_to_start(self, f):
        return self.map(f, lambda index, length: index == 0)

    @deprecated('0.19', 'use slicing instead')
    def apply_to_end(self, f):
        return self.map(f, lambda index, length: index == length - 1)

    @deprecated('0.19')
    def map(self, f, condition=None):
        if condition is None:
            condition = lambda index, length: True

        def iter_map_with_condition():
            length = len(self)
            for index, child in enumerate(self.parts):
                if hasattr(child, 'map'):
                    yield (child.map(f, condition) if condition(index, length) else child)
                else:
                    yield (f(child) if condition(index, length) else child)
        return self._create_similar(iter_map_with_condition())