import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _InstanceOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not isinstance(value, self.type):
            msg = "'{name}' must be {type!r} (got {value!r} that is a {actual!r}).".format(name=attr.name, type=self.type, actual=value.__class__, value=value)
            raise TypeError(msg, attr, self.type, value)

    def __repr__(self):
        return f'<instance_of validator for type {self.type!r}>'