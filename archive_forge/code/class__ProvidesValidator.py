import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _ProvidesValidator:
    interface = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.interface.providedBy(value):
            msg = "'{name}' must provide {interface!r} which {value!r} doesn't.".format(name=attr.name, interface=self.interface, value=value)
            raise TypeError(msg, attr, self.interface, value)

    def __repr__(self):
        return f'<provides validator for interface {self.interface!r}>'