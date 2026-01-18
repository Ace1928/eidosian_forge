import abc
import collections
import collections.abc
import operator
import sys
import typing
class _NotRequired(_MaybeRequired, _root=True):
    """A special typing construct to mark a key of a TypedDict as
        potentially missing. For example:

            class Movie(TypedDict):
                title: str
                year: NotRequired[int]

            m = Movie(
                title='The Matrix',  # typechecker error if key is omitted
                year=1999,
            )
        """