import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
class SchemaHelper(object):
    """IDL Schema helper.

    This class encapsulates the logic required to generate schemas suitable
    for creating 'ovs.db.idl.Idl' objects.  Clients should register columns
    they are interested in using register_columns().  When finished, the
    get_idl_schema() function may be called.

    The location on disk of the schema used may be found in the
    'schema_location' variable."""

    def __init__(self, location=None, schema_json=None):
        """Creates a new Schema object.

        'location' file path to ovs schema. None means default location
        'schema_json' schema in json preresentation in memory
        """
        if location and schema_json:
            raise ValueError("both location and schema_json can't be specified. it's ambiguous.")
        if schema_json is None:
            if location is None:
                location = '%s/vswitch.ovsschema' % ovs.dirs.PKGDATADIR
            schema_json = ovs.json.from_file(location)
        self.schema_json = schema_json
        self._tables = {}
        self._readonly = {}
        self._all = False

    def register_columns(self, table, columns, readonly=[]):
        """Registers interest in the given 'columns' of 'table'.  Future calls
        to get_idl_schema() will include 'table':column for each column in
        'columns'. This function automatically avoids adding duplicate entries
        to the schema.
        A subset of 'columns' can be specified as 'readonly'. The readonly
        columns are not replicated but can be fetched on-demand by the user
        with Row.fetch().

        'table' must be a string.
        'columns' must be a list of strings.
        'readonly' must be a list of strings.
        """
        assert isinstance(table, str)
        assert isinstance(columns, list)
        columns = set(columns) | self._tables.get(table, set())
        self._tables[table] = columns
        self._readonly[table] = readonly

    def register_table(self, table):
        """Registers interest in the given all columns of 'table'. Future calls
        to get_idl_schema() will include all columns of 'table'.

        'table' must be a string
        """
        assert isinstance(table, str)
        self._tables[table] = set()

    def register_all(self):
        """Registers interest in every column of every table."""
        self._all = True

    def get_idl_schema(self):
        """Gets a schema appropriate for the creation of an 'ovs.db.id.IDL'
        object based on columns registered using the register_columns()
        function."""
        schema = ovs.db.schema.DbSchema.from_json(self.schema_json)
        self.schema_json = None
        if not self._all:
            schema_tables = {}
            for table, columns in self._tables.items():
                schema_tables[table] = self._keep_table_columns(schema, table, columns)
            schema.tables = schema_tables
        schema.readonly = self._readonly
        return schema

    def _keep_table_columns(self, schema, table_name, columns):
        assert table_name in schema.tables
        table = schema.tables[table_name]
        if not columns:
            return table
        new_columns = {}
        for column_name in columns:
            assert isinstance(column_name, str)
            assert column_name in table.columns
            new_columns[column_name] = table.columns[column_name]
        table.columns = new_columns
        return table