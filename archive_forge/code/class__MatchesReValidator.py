import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, frozen=True, slots=True)
class _MatchesReValidator:
    pattern = attrib()
    match_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.match_func(value):
            msg = "'{name}' must match regex {pattern!r} ({value!r} doesn't)".format(name=attr.name, pattern=self.pattern.pattern, value=value)
            raise ValueError(msg, attr, self.pattern, value)

    def __repr__(self):
        return f'<matches_re validator for pattern {self.pattern!r}>'