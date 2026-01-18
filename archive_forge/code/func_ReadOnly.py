import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@_ExtensionsSpecialForm
def ReadOnly(self, parameters):
    """A special typing construct to mark an item of a TypedDict as read-only.

        For example:

            class Movie(TypedDict):
                title: ReadOnly[str]
                year: int

            def mutate_movie(m: Movie) -> None:
                m["year"] = 1992  # allowed
                m["title"] = "The Matrix"  # typechecker error

        There is no runtime checking for this property.
        """
    item = typing._type_check(parameters, f'{self._name} accepts only a single type.')
    return typing._GenericAlias(self, (item,))