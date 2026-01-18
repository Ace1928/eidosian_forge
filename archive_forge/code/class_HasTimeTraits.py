import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
class HasTimeTraits(HasStrictTraits):
    simple_time = Time()
    epoch = Time(UNIX_EPOCH)
    alternative_epoch = Time(default_value=NT_EPOCH)
    none_prohibited = Time(allow_none=False)
    none_allowed = Time(allow_none=True)