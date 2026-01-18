import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _NotValidator:
    validator = attrib()
    msg = attrib(converter=default_if_none("not_ validator child '{validator!r}' did not raise a captured error"))
    exc_types = attrib(validator=deep_iterable(member_validator=_subclass_of(Exception), iterable_validator=instance_of(tuple)))

    def __call__(self, inst, attr, value):
        try:
            self.validator(inst, attr, value)
        except self.exc_types:
            pass
        else:
            raise ValueError(self.msg.format(validator=self.validator, exc_types=self.exc_types), attr, self.validator, value, self.exc_types)

    def __repr__(self):
        return '<not_ validator wrapping {what!r}, capturing {exc_types!r}>'.format(what=self.validator, exc_types=self.exc_types)