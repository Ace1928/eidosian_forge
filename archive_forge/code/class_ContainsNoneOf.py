from __future__ import annotations
import re
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
from operator import attrgetter
from marshmallow import types
from marshmallow.exceptions import ValidationError
class ContainsNoneOf(NoneOf):
    """Validator which fails if ``value`` is a sequence and any element
    in the sequence is a member of the sequence passed as ``iterable``. Empty input
    is considered valid.

    :param iterable iterable: Same as :class:`NoneOf`.
    :param str error: Same as :class:`NoneOf`.

    .. versionadded:: 3.6.0
    """
    default_message = 'One or more of the choices you made was in: {values}.'

    def _format_error(self, value) -> str:
        value_text = ', '.join((str(val) for val in value))
        return super()._format_error(value_text)

    def __call__(self, value: typing.Sequence[_T]) -> typing.Sequence[_T]:
        for val in value:
            if val in self.iterable:
                raise ValidationError(self._format_error(value))
        return value