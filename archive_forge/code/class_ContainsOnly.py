from __future__ import annotations
import re
import typing
from abc import ABC, abstractmethod
from itertools import zip_longest
from operator import attrgetter
from marshmallow import types
from marshmallow.exceptions import ValidationError
class ContainsOnly(OneOf):
    """Validator which succeeds if ``value`` is a sequence and each element
    in the sequence is also in the sequence passed as ``choices``. Empty input
    is considered valid.

    :param iterable choices: Same as :class:`OneOf`.
    :param iterable labels: Same as :class:`OneOf`.
    :param str error: Same as :class:`OneOf`.

    .. versionchanged:: 3.0.0b2
        Duplicate values are considered valid.
    .. versionchanged:: 3.0.0b2
        Empty input is considered valid. Use `validate.Length(min=1) <marshmallow.validate.Length>`
        to validate against empty inputs.
    """
    default_message = 'One or more of the choices you made was not in: {choices}.'

    def _format_error(self, value) -> str:
        value_text = ', '.join((str(val) for val in value))
        return super()._format_error(value_text)

    def __call__(self, value: typing.Sequence[_T]) -> typing.Sequence[_T]:
        for val in value:
            if val not in self.choices:
                raise ValidationError(self._format_error(value))
        return value