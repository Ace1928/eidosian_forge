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
class HasSequenceTest(fixtures.TablesTest):
    run_deletes = None
    __requires__ = ('sequences',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        normalize_sequence(config, Sequence('user_id_seq', metadata=metadata))
        normalize_sequence(config, Sequence('other_seq', metadata=metadata, nomaxvalue=True, nominvalue=True))
        if testing.requires.schemas.enabled:
            normalize_sequence(config, Sequence('user_id_seq', schema=config.test_schema, metadata=metadata))
            normalize_sequence(config, Sequence('schema_seq', schema=config.test_schema, metadata=metadata))
        Table('user_id_table', metadata, Column('id', Integer, primary_key=True))

    def test_has_sequence(self, connection):
        eq_(inspect(connection).has_sequence('user_id_seq'), True)

    def test_has_sequence_cache(self, connection, metadata):
        insp = inspect(connection)
        eq_(insp.has_sequence('user_id_seq'), True)
        ss = normalize_sequence(config, Sequence('new_seq', metadata=metadata))
        eq_(insp.has_sequence('new_seq'), False)
        ss.create(connection)
        try:
            eq_(insp.has_sequence('new_seq'), False)
            insp.clear_cache()
            eq_(insp.has_sequence('new_seq'), True)
        finally:
            ss.drop(connection)

    def test_has_sequence_other_object(self, connection):
        eq_(inspect(connection).has_sequence('user_id_table'), False)

    @testing.requires.schemas
    def test_has_sequence_schema(self, connection):
        eq_(inspect(connection).has_sequence('user_id_seq', schema=config.test_schema), True)

    def test_has_sequence_neg(self, connection):
        eq_(inspect(connection).has_sequence('some_sequence'), False)

    @testing.requires.schemas
    def test_has_sequence_schemas_neg(self, connection):
        eq_(inspect(connection).has_sequence('some_sequence', schema=config.test_schema), False)

    @testing.requires.schemas
    def test_has_sequence_default_not_in_remote(self, connection):
        eq_(inspect(connection).has_sequence('other_sequence', schema=config.test_schema), False)

    @testing.requires.schemas
    def test_has_sequence_remote_not_in_default(self, connection):
        eq_(inspect(connection).has_sequence('schema_seq'), False)

    def test_get_sequence_names(self, connection):
        exp = {'other_seq', 'user_id_seq'}
        res = set(inspect(connection).get_sequence_names())
        is_true(res.intersection(exp) == exp)
        is_true('schema_seq' not in res)

    @testing.requires.schemas
    def test_get_sequence_names_no_sequence_schema(self, connection):
        eq_(inspect(connection).get_sequence_names(schema=config.test_schema_2), [])

    @testing.requires.schemas
    def test_get_sequence_names_sequences_schema(self, connection):
        eq_(sorted(inspect(connection).get_sequence_names(schema=config.test_schema)), ['schema_seq', 'user_id_seq'])