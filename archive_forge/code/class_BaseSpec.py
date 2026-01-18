import functools
import re
import warnings
class BaseSpec(object):
    """A specification of compatible versions.

    Usage:
    >>> Spec('>=1.0.0', syntax='npm')

    A version matches a specification if it matches any
    of the clauses of that specification.

    Internally, a Spec is AnyOf(
        AllOf(Matcher, Matcher, Matcher),
        AllOf(...),
    )
    """
    SYNTAXES = {}

    @classmethod
    def register_syntax(cls, subclass):
        syntax = subclass.SYNTAX
        if syntax is None:
            raise ValueError('A Spec needs its SYNTAX field to be set.')
        elif syntax in cls.SYNTAXES:
            raise ValueError('Duplicate syntax for %s: %r, %r' % (syntax, cls.SYNTAXES[syntax], subclass))
        cls.SYNTAXES[syntax] = subclass
        return subclass

    def __init__(self, expression):
        super(BaseSpec, self).__init__()
        self.expression = expression
        self.clause = self._parse_to_clause(expression)

    @classmethod
    def parse(cls, expression, syntax=DEFAULT_SYNTAX):
        """Convert a syntax-specific expression into a BaseSpec instance."""
        return cls.SYNTAXES[syntax](expression)

    @classmethod
    def _parse_to_clause(cls, expression):
        """Converts an expression to a clause."""
        raise NotImplementedError()

    def filter(self, versions):
        """Filter an iterable of versions satisfying the Spec."""
        for version in versions:
            if self.match(version):
                yield version

    def match(self, version):
        """Check whether a Version satisfies the Spec."""
        return self.clause.match(version)

    def select(self, versions):
        """Select the best compatible version among an iterable of options."""
        options = list(self.filter(versions))
        if options:
            return max(options)
        return None

    def __contains__(self, version):
        """Whether `version in self`."""
        if isinstance(version, Version):
            return self.match(version)
        return False

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.clause == other.clause

    def __hash__(self):
        return hash(self.clause)

    def __str__(self):
        return self.expression

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.expression)