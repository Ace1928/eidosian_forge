import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
def _subclass_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called
    with a wrong type for this particular attribute (checks are performed using
    `issubclass` therefore it's also valid to pass a tuple of types).

    :param type: The type to check for.
    :type type: type or tuple of types

    :raises TypeError: With a human readable error message, the attribute
        (of type `attrs.Attribute`), the expected type, and the value it
        got.
    """
    return _SubclassOfValidator(type)