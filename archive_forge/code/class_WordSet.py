import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
class WordSet(tuple):
    """
    Given an identifier, return the words that identifier represents,
    whether in camel case, underscore-separated, etc.

    >>> WordSet.parse("camelCase")
    ('camel', 'Case')

    >>> WordSet.parse("under_sep")
    ('under', 'sep')

    Acronyms should be retained

    >>> WordSet.parse("firstSNL")
    ('first', 'SNL')

    >>> WordSet.parse("you_and_I")
    ('you', 'and', 'I')

    >>> WordSet.parse("A simple test")
    ('A', 'simple', 'test')

    Multiple caps should not interfere with the first cap of another word.

    >>> WordSet.parse("myABCClass")
    ('my', 'ABC', 'Class')

    The result is a WordSet, providing access to other forms.

    >>> WordSet.parse("myABCClass").underscore_separated()
    'my_ABC_Class'

    >>> WordSet.parse('a-command').camel_case()
    'ACommand'

    >>> WordSet.parse('someIdentifier').lowered().space_separated()
    'some identifier'

    Slices of the result should return another WordSet.

    >>> WordSet.parse('taken-out-of-context')[1:].underscore_separated()
    'out_of_context'

    >>> WordSet.from_class_name(WordSet()).lowered().space_separated()
    'word set'

    >>> example = WordSet.parse('figured it out')
    >>> example.headless_camel_case()
    'figuredItOut'
    >>> example.dash_separated()
    'figured-it-out'

    """
    _pattern = re.compile('([A-Z]?[a-z]+)|([A-Z]+(?![a-z]))')

    def capitalized(self):
        return WordSet((word.capitalize() for word in self))

    def lowered(self):
        return WordSet((word.lower() for word in self))

    def camel_case(self):
        return ''.join(self.capitalized())

    def headless_camel_case(self):
        words = iter(self)
        first = next(words).lower()
        new_words = itertools.chain((first,), WordSet(words).camel_case())
        return ''.join(new_words)

    def underscore_separated(self):
        return '_'.join(self)

    def dash_separated(self):
        return '-'.join(self)

    def space_separated(self):
        return ' '.join(self)

    def trim_right(self, item):
        """
        Remove the item from the end of the set.

        >>> WordSet.parse('foo bar').trim_right('foo')
        ('foo', 'bar')
        >>> WordSet.parse('foo bar').trim_right('bar')
        ('foo',)
        >>> WordSet.parse('').trim_right('bar')
        ()
        """
        return self[:-1] if self and self[-1] == item else self

    def trim_left(self, item):
        """
        Remove the item from the beginning of the set.

        >>> WordSet.parse('foo bar').trim_left('foo')
        ('bar',)
        >>> WordSet.parse('foo bar').trim_left('bar')
        ('foo', 'bar')
        >>> WordSet.parse('').trim_left('bar')
        ()
        """
        return self[1:] if self and self[0] == item else self

    def trim(self, item):
        """
        >>> WordSet.parse('foo bar').trim('foo')
        ('bar',)
        """
        return self.trim_left(item).trim_right(item)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            result = WordSet(result)
        return result

    @classmethod
    def parse(cls, identifier):
        matches = cls._pattern.finditer(identifier)
        return WordSet((match.group(0) for match in matches))

    @classmethod
    def from_class_name(cls, subject):
        return cls.parse(subject.__class__.__name__)