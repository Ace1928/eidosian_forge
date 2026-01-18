import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
class DummyWeave:
    """A class that can be instantiated and compared."""

    def __init__(self, message):
        self._message = message
        self.finished = False

    def __eq__(self, other):
        if other is None:
            return False
        return self._message == other._message

    def __hash__(self):
        return hash((type(self), self._message))

    def transaction_finished(self):
        self.finished = True