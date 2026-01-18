import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Datetime, HasStrictTraits, TraitError
class HasDatetimeTraits(HasStrictTraits):
    simple_datetime = Datetime()
    epoch = Datetime(UNIX_EPOCH)
    alternative_epoch = Datetime(default_value=NT_EPOCH)
    none_prohibited = Datetime(allow_none=False)
    none_allowed = Datetime(allow_none=True)