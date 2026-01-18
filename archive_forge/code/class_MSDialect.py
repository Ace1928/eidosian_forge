from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
class MSDialect(default.DefaultDialect):
    name = 'mssql'
    supports_statement_cache = True
    supports_default_values = True
    supports_empty_insert = False
    favor_returning_over_lastrowid = True
    returns_native_bytes = True
    supports_comments = True
    supports_default_metavalue = False
    "dialect supports INSERT... VALUES (DEFAULT) syntax -\n    SQL Server **does** support this, but **not** for the IDENTITY column,\n    so we can't turn this on.\n\n    "
    execution_ctx_cls = MSExecutionContext
    use_scope_identity = True
    max_identifier_length = 128
    schema_name = 'dbo'
    insert_returning = True
    update_returning = True
    delete_returning = True
    update_returning_multifrom = True
    delete_returning_multifrom = True
    colspecs = {sqltypes.DateTime: _MSDateTime, sqltypes.Date: _MSDate, sqltypes.JSON: JSON, sqltypes.JSON.JSONIndexType: JSONIndexType, sqltypes.JSON.JSONPathType: JSONPathType, sqltypes.Time: _BASETIMEIMPL, sqltypes.Unicode: _MSUnicode, sqltypes.UnicodeText: _MSUnicodeText, DATETIMEOFFSET: DATETIMEOFFSET, DATETIME2: DATETIME2, SMALLDATETIME: SMALLDATETIME, DATETIME: DATETIME, sqltypes.Uuid: MSUUid}
    engine_config_types = default.DefaultDialect.engine_config_types.union({'legacy_schema_aliasing': util.asbool})
    ischema_names = ischema_names
    supports_sequences = True
    sequences_optional = True
    default_sequence_base = 1
    supports_native_boolean = False
    non_native_boolean_check_constraint = False
    supports_unicode_binds = True
    postfetch_lastrowid = True
    supports_multivalues_insert = True
    use_insertmanyvalues = True
    use_insertmanyvalues_wo_returning = True
    insertmanyvalues_implicit_sentinel = InsertmanyvaluesSentinelOpts.AUTOINCREMENT | InsertmanyvaluesSentinelOpts.IDENTITY | InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT
    insertmanyvalues_max_parameters = 2099
    _supports_offset_fetch = False
    _supports_nvarchar_max = False
    legacy_schema_aliasing = False
    server_version_info = ()
    statement_compiler = MSSQLCompiler
    ddl_compiler = MSDDLCompiler
    type_compiler_cls = MSTypeCompiler
    preparer = MSIdentifierPreparer
    construct_arguments = [(sa_schema.PrimaryKeyConstraint, {'clustered': None}), (sa_schema.UniqueConstraint, {'clustered': None}), (sa_schema.Index, {'clustered': None, 'include': None, 'where': None, 'columnstore': None}), (sa_schema.Column, {'identity_start': None, 'identity_increment': None})]

    def __init__(self, query_timeout=None, use_scope_identity=True, schema_name='dbo', deprecate_large_types=None, supports_comments=None, json_serializer=None, json_deserializer=None, legacy_schema_aliasing=None, ignore_no_transaction_on_rollback=False, **opts):
        self.query_timeout = int(query_timeout or 0)
        self.schema_name = schema_name
        self.use_scope_identity = use_scope_identity
        self.deprecate_large_types = deprecate_large_types
        self.ignore_no_transaction_on_rollback = ignore_no_transaction_on_rollback
        self._user_defined_supports_comments = uds = supports_comments
        if uds is not None:
            self.supports_comments = uds
        if legacy_schema_aliasing is not None:
            util.warn_deprecated('The legacy_schema_aliasing parameter is deprecated and will be removed in a future release.', '1.4')
            self.legacy_schema_aliasing = legacy_schema_aliasing
        super().__init__(**opts)
        self._json_serializer = json_serializer
        self._json_deserializer = json_deserializer

    def do_savepoint(self, connection, name):
        connection.exec_driver_sql('IF @@TRANCOUNT = 0 BEGIN TRANSACTION')
        super().do_savepoint(connection, name)

    def do_release_savepoint(self, connection, name):
        pass

    def do_rollback(self, dbapi_connection):
        try:
            super().do_rollback(dbapi_connection)
        except self.dbapi.ProgrammingError as e:
            if self.ignore_no_transaction_on_rollback and re.match('.*\\b111214\\b', str(e)):
                util.warn("ProgrammingError 111214 'No corresponding transaction found.' has been suppressed via ignore_no_transaction_on_rollback=True")
            else:
                raise
    _isolation_lookup = {'SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ', 'SNAPSHOT'}

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET TRANSACTION ISOLATION LEVEL {level}')
        cursor.close()
        if level == 'SNAPSHOT':
            dbapi_connection.commit()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        view_name = 'sys.system_views'
        try:
            cursor.execute("SELECT name FROM {} WHERE name IN ('dm_exec_sessions', 'dm_pdw_nodes_exec_sessions')".format(view_name))
            row = cursor.fetchone()
            if not row:
                raise NotImplementedError("Can't fetch isolation level on this particular SQL Server version.")
            view_name = f'sys.{row[0]}'
            cursor.execute("\n                    SELECT CASE transaction_isolation_level\n                    WHEN 0 THEN NULL\n                    WHEN 1 THEN 'READ UNCOMMITTED'\n                    WHEN 2 THEN 'READ COMMITTED'\n                    WHEN 3 THEN 'REPEATABLE READ'\n                    WHEN 4 THEN 'SERIALIZABLE'\n                    WHEN 5 THEN 'SNAPSHOT' END\n                    AS TRANSACTION_ISOLATION_LEVEL\n                    FROM {}\n                    where session_id = @@SPID\n                ".format(view_name))
        except self.dbapi.Error as err:
            raise NotImplementedError('Can\'t fetch isolation level;  encountered error {} when attempting to query the "{}" view.'.format(err, view_name)) from err
        else:
            row = cursor.fetchone()
            return row[0].upper()
        finally:
            cursor.close()

    def initialize(self, connection):
        super().initialize(connection)
        self._setup_version_attributes()
        self._setup_supports_nvarchar_max(connection)
        self._setup_supports_comments(connection)

    def _setup_version_attributes(self):
        if self.server_version_info[0] not in list(range(8, 17)):
            util.warn("Unrecognized server version info '%s'.  Some SQL Server features may not function properly." % '.'.join((str(x) for x in self.server_version_info)))
        if self.server_version_info >= MS_2008_VERSION:
            self.supports_multivalues_insert = True
        else:
            self.supports_multivalues_insert = False
        if self.deprecate_large_types is None:
            self.deprecate_large_types = self.server_version_info >= MS_2012_VERSION
        self._supports_offset_fetch = self.server_version_info and self.server_version_info[0] >= 11

    def _setup_supports_nvarchar_max(self, connection):
        try:
            connection.scalar(sql.text("SELECT CAST('test max support' AS NVARCHAR(max))"))
        except exc.DBAPIError:
            self._supports_nvarchar_max = False
        else:
            self._supports_nvarchar_max = True

    def _setup_supports_comments(self, connection):
        if self._user_defined_supports_comments is not None:
            return
        try:
            connection.scalar(sql.text('SELECT 1 FROM fn_listextendedproperty(default, default, default, default, default, default, default)'))
        except exc.DBAPIError:
            self.supports_comments = False
        else:
            self.supports_comments = True

    def _get_default_schema_name(self, connection):
        query = sql.text('SELECT schema_name()')
        default_schema_name = connection.scalar(query)
        if default_schema_name is not None:
            return quoted_name(default_schema_name, quote=True)
        else:
            return self.schema_name

    @_db_plus_owner
    def has_table(self, connection, tablename, dbname, owner, schema, **kw):
        self._ensure_has_table_connection(connection)
        return self._internal_has_table(connection, tablename, owner, **kw)

    @reflection.cache
    @_db_plus_owner
    def has_sequence(self, connection, sequencename, dbname, owner, schema, **kw):
        sequences = ischema.sequences
        s = sql.select(sequences.c.sequence_name).where(sequences.c.sequence_name == sequencename)
        if owner:
            s = s.where(sequences.c.sequence_schema == owner)
        c = connection.execute(s)
        return c.first() is not None

    @reflection.cache
    @_db_plus_owner_listing
    def get_sequence_names(self, connection, dbname, owner, schema, **kw):
        sequences = ischema.sequences
        s = sql.select(sequences.c.sequence_name)
        if owner:
            s = s.where(sequences.c.sequence_schema == owner)
        c = connection.execute(s)
        return [row[0] for row in c]

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        s = sql.select(ischema.schemata.c.schema_name).order_by(ischema.schemata.c.schema_name)
        schema_names = [r[0] for r in connection.execute(s)]
        return schema_names

    @reflection.cache
    @_db_plus_owner_listing
    def get_table_names(self, connection, dbname, owner, schema, **kw):
        tables = ischema.tables
        s = sql.select(tables.c.table_name).where(sql.and_(tables.c.table_schema == owner, tables.c.table_type == 'BASE TABLE')).order_by(tables.c.table_name)
        table_names = [r[0] for r in connection.execute(s)]
        return table_names

    @reflection.cache
    @_db_plus_owner_listing
    def get_view_names(self, connection, dbname, owner, schema, **kw):
        tables = ischema.tables
        s = sql.select(tables.c.table_name).where(sql.and_(tables.c.table_schema == owner, tables.c.table_type == 'VIEW')).order_by(tables.c.table_name)
        view_names = [r[0] for r in connection.execute(s)]
        return view_names

    @reflection.cache
    def _internal_has_table(self, connection, tablename, owner, **kw):
        if tablename.startswith('#'):
            return bool(connection.scalar(text("SELECT object_id(:table_name, 'U')"), {'table_name': f'tempdb.dbo.[{tablename}]'}))
        else:
            tables = ischema.tables
            s = sql.select(tables.c.table_name).where(sql.and_(sql.or_(tables.c.table_type == 'BASE TABLE', tables.c.table_type == 'VIEW'), tables.c.table_name == tablename))
            if owner:
                s = s.where(tables.c.table_schema == owner)
            c = connection.execute(s)
            return c.first() is not None

    def _default_or_error(self, connection, tablename, owner, method, **kw):
        if self._internal_has_table(connection, tablename, owner, **kw):
            return method()
        else:
            raise exc.NoSuchTableError(f'{owner}.{tablename}')

    @reflection.cache
    @_db_plus_owner
    def get_indexes(self, connection, tablename, dbname, owner, schema, **kw):
        filter_definition = 'ind.filter_definition' if self.server_version_info >= MS_2008_VERSION else 'NULL as filter_definition'
        rp = connection.execution_options(future_result=True).execute(sql.text(f'\nselect\n    ind.index_id,\n    ind.is_unique,\n    ind.name,\n    ind.type,\n    {filter_definition}\nfrom\n    sys.indexes as ind\njoin sys.tables as tab on\n    ind.object_id = tab.object_id\njoin sys.schemas as sch on\n    sch.schema_id = tab.schema_id\nwhere\n    tab.name = :tabname\n    and sch.name = :schname\n    and ind.is_primary_key = 0\n    and ind.type != 0\norder by\n    ind.name\n                ').bindparams(sql.bindparam('tabname', tablename, ischema.CoerceUnicode()), sql.bindparam('schname', owner, ischema.CoerceUnicode())).columns(name=sqltypes.Unicode()))
        indexes = {}
        for row in rp.mappings():
            indexes[row['index_id']] = current = {'name': row['name'], 'unique': row['is_unique'] == 1, 'column_names': [], 'include_columns': [], 'dialect_options': {}}
            do = current['dialect_options']
            index_type = row['type']
            if index_type in {1, 2}:
                do['mssql_clustered'] = index_type == 1
            if index_type in {5, 6}:
                do['mssql_clustered'] = index_type == 5
                do['mssql_columnstore'] = True
            if row['filter_definition'] is not None:
                do['mssql_where'] = row['filter_definition']
        rp = connection.execution_options(future_result=True).execute(sql.text('\nselect\n    ind_col.index_id,\n    col.name,\n    ind_col.is_included_column\nfrom\n    sys.columns as col\njoin sys.tables as tab on\n    tab.object_id = col.object_id\njoin sys.index_columns as ind_col on\n    ind_col.column_id = col.column_id\n    and ind_col.object_id = tab.object_id\njoin sys.schemas as sch on\n    sch.schema_id = tab.schema_id\nwhere\n    tab.name = :tabname\n    and sch.name = :schname\n            ').bindparams(sql.bindparam('tabname', tablename, ischema.CoerceUnicode()), sql.bindparam('schname', owner, ischema.CoerceUnicode())).columns(name=sqltypes.Unicode()))
        for row in rp.mappings():
            if row['index_id'] not in indexes:
                continue
            index_def = indexes[row['index_id']]
            is_colstore = index_def['dialect_options'].get('mssql_columnstore')
            is_clustered = index_def['dialect_options'].get('mssql_clustered')
            if not (is_colstore and is_clustered):
                if row['is_included_column'] and (not is_colstore):
                    index_def['include_columns'].append(row['name'])
                else:
                    index_def['column_names'].append(row['name'])
        for index_info in indexes.values():
            index_info['dialect_options']['mssql_include'] = index_info['include_columns']
        if indexes:
            return list(indexes.values())
        else:
            return self._default_or_error(connection, tablename, owner, ReflectionDefaults.indexes, **kw)

    @reflection.cache
    @_db_plus_owner
    def get_view_definition(self, connection, viewname, dbname, owner, schema, **kw):
        view_def = connection.execute(sql.text('select mod.definition from sys.sql_modules as mod join sys.views as views on mod.object_id = views.object_id join sys.schemas as sch on views.schema_id = sch.schema_id where views.name=:viewname and sch.name=:schname').bindparams(sql.bindparam('viewname', viewname, ischema.CoerceUnicode()), sql.bindparam('schname', owner, ischema.CoerceUnicode()))).scalar()
        if view_def:
            return view_def
        else:
            raise exc.NoSuchTableError(f'{owner}.{viewname}')

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        if not self.supports_comments:
            raise NotImplementedError("Can't get table comments on current SQL Server version in use")
        schema_name = schema if schema else self.default_schema_name
        COMMENT_SQL = "\n            SELECT cast(com.value as nvarchar(max))\n            FROM fn_listextendedproperty('MS_Description',\n                'schema', :schema, 'table', :table, NULL, NULL\n            ) as com;\n        "
        comment = connection.execute(sql.text(COMMENT_SQL).bindparams(sql.bindparam('schema', schema_name, ischema.CoerceUnicode()), sql.bindparam('table', table_name, ischema.CoerceUnicode()))).scalar()
        if comment:
            return {'text': comment}
        else:
            return self._default_or_error(connection, table_name, None, ReflectionDefaults.table_comment, **kw)

    def _temp_table_name_like_pattern(self, tablename):
        return tablename + ('[_][_][_]%' if not tablename.startswith('##') else '')

    def _get_internal_temp_table_name(self, connection, tablename):
        try:
            return connection.execute(sql.text('select table_schema, table_name from tempdb.information_schema.tables where table_name like :p1'), {'p1': self._temp_table_name_like_pattern(tablename)}).one()
        except exc.MultipleResultsFound as me:
            raise exc.UnreflectableTableError("Found more than one temporary table named '%s' in tempdb at this time. Cannot reliably resolve that name to its internal table name." % tablename) from me
        except exc.NoResultFound as ne:
            raise exc.NoSuchTableError("Unable to find a temporary table named '%s' in tempdb." % tablename) from ne

    @reflection.cache
    @_db_plus_owner
    def get_columns(self, connection, tablename, dbname, owner, schema, **kw):
        is_temp_table = tablename.startswith('#')
        if is_temp_table:
            owner, tablename = self._get_internal_temp_table_name(connection, tablename)
            columns = ischema.mssql_temp_table_columns
        else:
            columns = ischema.columns
        computed_cols = ischema.computed_columns
        identity_cols = ischema.identity_columns
        if owner:
            whereclause = sql.and_(columns.c.table_name == tablename, columns.c.table_schema == owner)
            full_name = columns.c.table_schema + '.' + columns.c.table_name
        else:
            whereclause = columns.c.table_name == tablename
            full_name = columns.c.table_name
        if self._supports_nvarchar_max:
            computed_definition = computed_cols.c.definition
        else:
            computed_definition = sql.cast(computed_cols.c.definition, NVARCHAR(4000))
        object_id = func.object_id(full_name)
        s = sql.select(columns.c.column_name, columns.c.data_type, columns.c.is_nullable, columns.c.character_maximum_length, columns.c.numeric_precision, columns.c.numeric_scale, columns.c.column_default, columns.c.collation_name, computed_definition, computed_cols.c.is_persisted, identity_cols.c.is_identity, identity_cols.c.seed_value, identity_cols.c.increment_value, ischema.extended_properties.c.value.label('comment')).select_from(columns).outerjoin(computed_cols, onclause=sql.and_(computed_cols.c.object_id == object_id, computed_cols.c.name == columns.c.column_name.collate('DATABASE_DEFAULT'))).outerjoin(identity_cols, onclause=sql.and_(identity_cols.c.object_id == object_id, identity_cols.c.name == columns.c.column_name.collate('DATABASE_DEFAULT'))).outerjoin(ischema.extended_properties, onclause=sql.and_(ischema.extended_properties.c['class'] == 1, ischema.extended_properties.c.major_id == object_id, ischema.extended_properties.c.minor_id == columns.c.ordinal_position, ischema.extended_properties.c.name == 'MS_Description')).where(whereclause).order_by(columns.c.ordinal_position)
        c = connection.execution_options(future_result=True).execute(s)
        cols = []
        for row in c.mappings():
            name = row[columns.c.column_name]
            type_ = row[columns.c.data_type]
            nullable = row[columns.c.is_nullable] == 'YES'
            charlen = row[columns.c.character_maximum_length]
            numericprec = row[columns.c.numeric_precision]
            numericscale = row[columns.c.numeric_scale]
            default = row[columns.c.column_default]
            collation = row[columns.c.collation_name]
            definition = row[computed_definition]
            is_persisted = row[computed_cols.c.is_persisted]
            is_identity = row[identity_cols.c.is_identity]
            identity_start = row[identity_cols.c.seed_value]
            identity_increment = row[identity_cols.c.increment_value]
            comment = row[ischema.extended_properties.c.value]
            coltype = self.ischema_names.get(type_, None)
            kwargs = {}
            if coltype in (MSString, MSChar, MSNVarchar, MSNChar, MSText, MSNText, MSBinary, MSVarBinary, sqltypes.LargeBinary):
                if charlen == -1:
                    charlen = None
                kwargs['length'] = charlen
                if collation:
                    kwargs['collation'] = collation
            if coltype is None:
                util.warn("Did not recognize type '%s' of column '%s'" % (type_, name))
                coltype = sqltypes.NULLTYPE
            else:
                if issubclass(coltype, sqltypes.Numeric):
                    kwargs['precision'] = numericprec
                    if not issubclass(coltype, sqltypes.Float):
                        kwargs['scale'] = numericscale
                coltype = coltype(**kwargs)
            cdict = {'name': name, 'type': coltype, 'nullable': nullable, 'default': default, 'autoincrement': is_identity is not None, 'comment': comment}
            if definition is not None and is_persisted is not None:
                cdict['computed'] = {'sqltext': definition, 'persisted': is_persisted}
            if is_identity is not None:
                if identity_start is None or identity_increment is None:
                    cdict['identity'] = {}
                else:
                    if isinstance(coltype, sqltypes.BigInteger):
                        start = int(identity_start)
                        increment = int(identity_increment)
                    elif isinstance(coltype, sqltypes.Integer):
                        start = int(identity_start)
                        increment = int(identity_increment)
                    else:
                        start = identity_start
                        increment = identity_increment
                    cdict['identity'] = {'start': start, 'increment': increment}
            cols.append(cdict)
        if cols:
            return cols
        else:
            return self._default_or_error(connection, tablename, owner, ReflectionDefaults.columns, **kw)

    @reflection.cache
    @_db_plus_owner
    def get_pk_constraint(self, connection, tablename, dbname, owner, schema, **kw):
        pkeys = []
        TC = ischema.constraints
        C = ischema.key_constraints.alias('C')
        s = sql.select(C.c.column_name, TC.c.constraint_type, C.c.constraint_name, func.objectproperty(func.object_id(C.c.table_schema + '.' + C.c.constraint_name), 'CnstIsClustKey').label('is_clustered')).where(sql.and_(TC.c.constraint_name == C.c.constraint_name, TC.c.table_schema == C.c.table_schema, C.c.table_name == tablename, C.c.table_schema == owner)).order_by(TC.c.constraint_name, C.c.ordinal_position)
        c = connection.execution_options(future_result=True).execute(s)
        constraint_name = None
        is_clustered = None
        for row in c.mappings():
            if 'PRIMARY' in row[TC.c.constraint_type.name]:
                pkeys.append(row['COLUMN_NAME'])
                if constraint_name is None:
                    constraint_name = row[C.c.constraint_name.name]
                if is_clustered is None:
                    is_clustered = row['is_clustered']
        if pkeys:
            return {'constrained_columns': pkeys, 'name': constraint_name, 'dialect_options': {'mssql_clustered': is_clustered}}
        else:
            return self._default_or_error(connection, tablename, owner, ReflectionDefaults.pk_constraint, **kw)

    @reflection.cache
    @_db_plus_owner
    def get_foreign_keys(self, connection, tablename, dbname, owner, schema, **kw):
        s = text('WITH fk_info AS (\n    SELECT\n        ischema_ref_con.constraint_schema,\n        ischema_ref_con.constraint_name,\n        ischema_key_col.ordinal_position,\n        ischema_key_col.table_schema,\n        ischema_key_col.table_name,\n        ischema_ref_con.unique_constraint_schema,\n        ischema_ref_con.unique_constraint_name,\n        ischema_ref_con.match_option,\n        ischema_ref_con.update_rule,\n        ischema_ref_con.delete_rule,\n        ischema_key_col.column_name AS constrained_column\n    FROM\n        INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS ischema_ref_con\n        INNER JOIN\n        INFORMATION_SCHEMA.KEY_COLUMN_USAGE ischema_key_col ON\n            ischema_key_col.table_schema = ischema_ref_con.constraint_schema\n            AND ischema_key_col.constraint_name =\n            ischema_ref_con.constraint_name\n    WHERE ischema_key_col.table_name = :tablename\n        AND ischema_key_col.table_schema = :owner\n),\nconstraint_info AS (\n    SELECT\n        ischema_key_col.constraint_schema,\n        ischema_key_col.constraint_name,\n        ischema_key_col.ordinal_position,\n        ischema_key_col.table_schema,\n        ischema_key_col.table_name,\n        ischema_key_col.column_name\n    FROM\n        INFORMATION_SCHEMA.KEY_COLUMN_USAGE ischema_key_col\n),\nindex_info AS (\n    SELECT\n        sys.schemas.name AS index_schema,\n        sys.indexes.name AS index_name,\n        sys.index_columns.key_ordinal AS ordinal_position,\n        sys.schemas.name AS table_schema,\n        sys.objects.name AS table_name,\n        sys.columns.name AS column_name\n    FROM\n        sys.indexes\n        INNER JOIN\n        sys.objects ON\n            sys.objects.object_id = sys.indexes.object_id\n        INNER JOIN\n        sys.schemas ON\n            sys.schemas.schema_id = sys.objects.schema_id\n        INNER JOIN\n        sys.index_columns ON\n            sys.index_columns.object_id = sys.objects.object_id\n            AND sys.index_columns.index_id = sys.indexes.index_id\n        INNER JOIN\n        sys.columns ON\n            sys.columns.object_id = sys.indexes.object_id\n            AND sys.columns.column_id = sys.index_columns.column_id\n)\n    SELECT\n        fk_info.constraint_schema,\n        fk_info.constraint_name,\n        fk_info.ordinal_position,\n        fk_info.constrained_column,\n        constraint_info.table_schema AS referred_table_schema,\n        constraint_info.table_name AS referred_table_name,\n        constraint_info.column_name AS referred_column,\n        fk_info.match_option,\n        fk_info.update_rule,\n        fk_info.delete_rule\n    FROM\n        fk_info INNER JOIN constraint_info ON\n            constraint_info.constraint_schema =\n                fk_info.unique_constraint_schema\n            AND constraint_info.constraint_name =\n                fk_info.unique_constraint_name\n            AND constraint_info.ordinal_position = fk_info.ordinal_position\n    UNION\n    SELECT\n        fk_info.constraint_schema,\n        fk_info.constraint_name,\n        fk_info.ordinal_position,\n        fk_info.constrained_column,\n        index_info.table_schema AS referred_table_schema,\n        index_info.table_name AS referred_table_name,\n        index_info.column_name AS referred_column,\n        fk_info.match_option,\n        fk_info.update_rule,\n        fk_info.delete_rule\n    FROM\n        fk_info INNER JOIN index_info ON\n            index_info.index_schema = fk_info.unique_constraint_schema\n            AND index_info.index_name = fk_info.unique_constraint_name\n            AND index_info.ordinal_position = fk_info.ordinal_position\n\n    ORDER BY fk_info.constraint_schema, fk_info.constraint_name,\n        fk_info.ordinal_position\n').bindparams(sql.bindparam('tablename', tablename, ischema.CoerceUnicode()), sql.bindparam('owner', owner, ischema.CoerceUnicode())).columns(constraint_schema=sqltypes.Unicode(), constraint_name=sqltypes.Unicode(), table_schema=sqltypes.Unicode(), table_name=sqltypes.Unicode(), constrained_column=sqltypes.Unicode(), referred_table_schema=sqltypes.Unicode(), referred_table_name=sqltypes.Unicode(), referred_column=sqltypes.Unicode())
        fkeys = []

        def fkey_rec():
            return {'name': None, 'constrained_columns': [], 'referred_schema': None, 'referred_table': None, 'referred_columns': [], 'options': {}}
        fkeys = util.defaultdict(fkey_rec)
        for r in connection.execute(s).all():
            _, rfknm, _, scol, rschema, rtbl, rcol, _, fkuprule, fkdelrule = r
            rec = fkeys[rfknm]
            rec['name'] = rfknm
            if fkuprule != 'NO ACTION':
                rec['options']['onupdate'] = fkuprule
            if fkdelrule != 'NO ACTION':
                rec['options']['ondelete'] = fkdelrule
            if not rec['referred_table']:
                rec['referred_table'] = rtbl
                if schema is not None or owner != rschema:
                    if dbname:
                        rschema = dbname + '.' + rschema
                    rec['referred_schema'] = rschema
            local_cols, remote_cols = (rec['constrained_columns'], rec['referred_columns'])
            local_cols.append(scol)
            remote_cols.append(rcol)
        if fkeys:
            return list(fkeys.values())
        else:
            return self._default_or_error(connection, tablename, owner, ReflectionDefaults.foreign_keys, **kw)