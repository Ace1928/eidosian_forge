from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
class PluralRule:
    """Represents a set of language pluralization rules.  The constructor
    accepts a list of (tag, expr) tuples or a dict of `CLDR rules`_. The
    resulting object is callable and accepts one parameter with a positive or
    negative number (both integer and float) for the number that indicates the
    plural form for a string and returns the tag for the format:

    >>> rule = PluralRule({'one': 'n is 1'})
    >>> rule(1)
    'one'
    >>> rule(2)
    'other'

    Currently the CLDR defines these tags: zero, one, two, few, many and
    other where other is an implicit default.  Rules should be mutually
    exclusive; for a given numeric value, only one rule should apply (i.e.
    the condition should only be true for one of the plural rule elements.

    .. _`CLDR rules`: https://www.unicode.org/reports/tr35/tr35-33/tr35-numbers.html#Language_Plural_Rules
    """
    __slots__ = ('abstract', '_func')

    def __init__(self, rules: Mapping[str, str] | Iterable[tuple[str, str]]) -> None:
        """Initialize the rule instance.

        :param rules: a list of ``(tag, expr)``) tuples with the rules
                      conforming to UTS #35 or a dict with the tags as keys
                      and expressions as values.
        :raise RuleError: if the expression is malformed
        """
        if isinstance(rules, Mapping):
            rules = rules.items()
        found = set()
        self.abstract: list[tuple[str, Any]] = []
        for key, expr in sorted(rules):
            if key not in _plural_tags:
                raise ValueError(f'unknown tag {key!r}')
            elif key in found:
                raise ValueError(f'tag {key!r} defined twice')
            found.add(key)
            ast = _Parser(expr).ast
            if ast:
                self.abstract.append((key, ast))

    def __repr__(self) -> str:
        rules = self.rules
        args = ', '.join([f'{tag}: {rules[tag]}' for tag in _plural_tags if tag in rules])
        return f'<{type(self).__name__} {args!r}>'

    @classmethod
    def parse(cls, rules: Mapping[str, str] | Iterable[tuple[str, str]] | PluralRule) -> PluralRule:
        """Create a `PluralRule` instance for the given rules.  If the rules
        are a `PluralRule` object, that object is returned.

        :param rules: the rules as list or dict, or a `PluralRule` object
        :raise RuleError: if the expression is malformed
        """
        if isinstance(rules, PluralRule):
            return rules
        return cls(rules)

    @property
    def rules(self) -> Mapping[str, str]:
        """The `PluralRule` as a dict of unicode plural rules.

        >>> rule = PluralRule({'one': 'n is 1'})
        >>> rule.rules
        {'one': 'n is 1'}
        """
        _compile = _UnicodeCompiler().compile
        return {tag: _compile(ast) for tag, ast in self.abstract}

    @property
    def tags(self) -> frozenset[str]:
        """A set of explicitly defined tags in this rule.  The implicit default
        ``'other'`` rules is not part of this set unless there is an explicit
        rule for it.
        """
        return frozenset((i[0] for i in self.abstract))

    def __getstate__(self) -> list[tuple[str, Any]]:
        return self.abstract

    def __setstate__(self, abstract: list[tuple[str, Any]]) -> None:
        self.abstract = abstract

    def __call__(self, n: float | decimal.Decimal) -> str:
        if not hasattr(self, '_func'):
            self._func = to_python(self)
        return self._func(n)