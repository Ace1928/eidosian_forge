from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
class MatchesStructure:
    """Matcher that matches an object structurally.

    'Structurally' here means that attributes of the object being matched are
    compared against given matchers.

    `fromExample` allows the creation of a matcher from a prototype object and
    then modified versions can be created with `update`.

    `byEquality` creates a matcher in much the same way as the constructor,
    except that the matcher for each of the attributes is assumed to be
    `Equals`.

    `byMatcher` creates a similar matcher to `byEquality`, but you get to pick
    the matcher, rather than just using `Equals`.
    """

    def __init__(self, **kwargs):
        """Construct a `MatchesStructure`.

        :param kwargs: A mapping of attributes to matchers.
        """
        self.kws = kwargs

    @classmethod
    def byEquality(cls, **kwargs):
        """Matches an object where the attributes equal the keyword values.

        Similar to the constructor, except that the matcher is assumed to be
        Equals.
        """
        from ._basic import Equals
        return cls.byMatcher(Equals, **kwargs)

    @classmethod
    def byMatcher(cls, matcher, **kwargs):
        """Matches an object where the attributes match the keyword values.

        Similar to the constructor, except that the provided matcher is used
        to match all of the values.
        """
        return cls(**map_values(matcher, kwargs))

    @classmethod
    def fromExample(cls, example, *attributes):
        from ._basic import Equals
        kwargs = {}
        for attr in attributes:
            kwargs[attr] = Equals(getattr(example, attr))
        return cls(**kwargs)

    def update(self, **kws):
        new_kws = self.kws.copy()
        for attr, matcher in kws.items():
            if matcher is None:
                new_kws.pop(attr, None)
            else:
                new_kws[attr] = matcher
        return type(self)(**new_kws)

    def __str__(self):
        kws = []
        for attr, matcher in sorted(self.kws.items()):
            kws.append(f'{attr}={matcher}')
        return '{}({})'.format(self.__class__.__name__, ', '.join(kws))

    def match(self, value):
        matchers = []
        values = []
        for attr, matcher in sorted(self.kws.items()):
            matchers.append(Annotate(attr, matcher))
            values.append(getattr(value, attr))
        return MatchesListwise(matchers).match(values)