import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _OptionalValidator:
    validator = attrib()

    def __call__(self, inst, attr, value):
        if value is None:
            return
        self.validator(inst, attr, value)

    def __repr__(self):
        return f'<optional validator for {self.validator!r} or None>'