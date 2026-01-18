import unittest
from traits.api import (
from traits.tests.tuple_test_mixin import TupleTestMixin
class BadInt(BaseInt):
    """ Test class used to simulate a Tuple item with bad validation.
    """
    info_text = 'a bad integer'

    def validate(self, object, name, value):
        return 1 / 0