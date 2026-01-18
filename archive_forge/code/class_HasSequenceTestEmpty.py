from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing
class HasSequenceTestEmpty(fixtures.TestBase):
    __requires__ = ('sequences',)
    __backend__ = True

    def test_get_sequence_names_no_sequence(self, connection):
        eq_(inspect(connection).get_sequence_names(), [])