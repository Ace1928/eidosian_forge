import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Date, HasStrictTraits, TraitError
class HasDateTraits(HasStrictTraits):
    simple_date = Date()
    epoch = Date(UNIX_EPOCH)
    alternative_epoch = Date(default_value=NT_EPOCH)
    datetime_prohibited = Date(allow_datetime=False)
    datetime_allowed = Date(allow_datetime=True)
    none_prohibited = Date(allow_none=False)
    none_allowed = Date(allow_none=True)
    strict = Date(allow_datetime=False, allow_none=False)