from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
class SQLiteDialect(default.DefaultDialect):
    name = 'sqlite'
    supports_alter = False
    supports_default_values = True
    supports_default_metavalue = False
    supports_sane_rowcount_returning = False
    supports_empty_insert = False
    supports_cast = True
    supports_multivalues_insert = True
    use_insertmanyvalues = True
    tuple_in_values = True
    supports_statement_cache = True
    insert_null_pk_still_autoincrements = True
    insert_returning = True
    update_returning = True
    update_returning_multifrom = True
    delete_returning = True
    update_returning_multifrom = True
    supports_default_metavalue = True
    'dialect supports INSERT... VALUES (DEFAULT) syntax'
    default_metavalue_token = 'NULL'
    'for INSERT... VALUES (DEFAULT) syntax, the token to put in the\n    parenthesis.'
    default_paramstyle = 'qmark'
    execution_ctx_cls = SQLiteExecutionContext
    statement_compiler = SQLiteCompiler
    ddl_compiler = SQLiteDDLCompiler
    type_compiler_cls = SQLiteTypeCompiler
    preparer = SQLiteIdentifierPreparer
    ischema_names = ischema_names
    colspecs = colspecs
    construct_arguments = [(sa_schema.Table, {'autoincrement': False, 'with_rowid': True}), (sa_schema.Index, {'where': None}), (sa_schema.Column, {'on_conflict_primary_key': None, 'on_conflict_not_null': None, 'on_conflict_unique': None}), (sa_schema.Constraint, {'on_conflict': None})]
    _broken_fk_pragma_quotes = False
    _broken_dotted_colnames = False

    @util.deprecated_params(_json_serializer=('1.3.7', 'The _json_serializer argument to the SQLite dialect has been renamed to the correct name of json_serializer.  The old argument name will be removed in a future release.'), _json_deserializer=('1.3.7', 'The _json_deserializer argument to the SQLite dialect has been renamed to the correct name of json_deserializer.  The old argument name will be removed in a future release.'))
    def __init__(self, native_datetime=False, json_serializer=None, json_deserializer=None, _json_serializer=None, _json_deserializer=None, **kwargs):
        default.DefaultDialect.__init__(self, **kwargs)
        if _json_serializer:
            json_serializer = _json_serializer
        if _json_deserializer:
            json_deserializer = _json_deserializer
        self._json_serializer = json_serializer
        self._json_deserializer = json_deserializer
        self.native_datetime = native_datetime
        if self.dbapi is not None:
            if self.dbapi.sqlite_version_info < (3, 7, 16):
                util.warn('SQLite version %s is older than 3.7.16, and will not support right nested joins, as are sometimes used in more complex ORM scenarios.  SQLAlchemy 1.4 and above no longer tries to rewrite these joins.' % (self.dbapi.sqlite_version_info,))
            self._broken_dotted_colnames = self.dbapi.sqlite_version_info < (3, 10, 0)
            self.supports_default_values = self.dbapi.sqlite_version_info >= (3, 3, 8)
            self.supports_cast = self.dbapi.sqlite_version_info >= (3, 2, 3)
            self.supports_multivalues_insert = self.dbapi.sqlite_version_info >= (3, 7, 11)
            self._broken_fk_pragma_quotes = self.dbapi.sqlite_version_info < (3, 6, 14)
            if self.dbapi.sqlite_version_info < (3, 35) or util.pypy:
                self.update_returning = self.delete_returning = self.insert_returning = False
            if self.dbapi.sqlite_version_info < (3, 32, 0):
                self.insertmanyvalues_max_parameters = 999
    _isolation_lookup = util.immutabledict({'READ UNCOMMITTED': 1, 'SERIALIZABLE': 0})

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        isolation_level = self._isolation_lookup[level]
        cursor = dbapi_connection.cursor()
        cursor.execute(f'PRAGMA read_uncommitted = {isolation_level}')
        cursor.close()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA read_uncommitted')
        res = cursor.fetchone()
        if res:
            value = res[0]
        else:
            value = 0
        cursor.close()
        if value == 0:
            return 'SERIALIZABLE'
        elif value == 1:
            return 'READ UNCOMMITTED'
        else:
            assert False, 'Unknown isolation level %s' % value

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        s = 'PRAGMA database_list'
        dl = connection.exec_driver_sql(s)
        return [db[1] for db in dl if db[1] != 'temp']

    def _format_schema(self, schema, table_name):
        if schema is not None:
            qschema = self.identifier_preparer.quote_identifier(schema)
            name = f'{qschema}.{table_name}'
        else:
            name = table_name
        return name

    def _sqlite_main_query(self, table: str, type_: str, schema: Optional[str], sqlite_include_internal: bool):
        main = self._format_schema(schema, table)
        if not sqlite_include_internal:
            filter_table = " AND name NOT LIKE 'sqlite~_%' ESCAPE '~'"
        else:
            filter_table = ''
        query = f"SELECT name FROM {main} WHERE type='{type_}'{filter_table} ORDER BY name"
        return query

    @reflection.cache
    def get_table_names(self, connection, schema=None, sqlite_include_internal=False, **kw):
        query = self._sqlite_main_query('sqlite_master', 'table', schema, sqlite_include_internal)
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_temp_table_names(self, connection, sqlite_include_internal=False, **kw):
        query = self._sqlite_main_query('sqlite_temp_master', 'table', None, sqlite_include_internal)
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_temp_view_names(self, connection, sqlite_include_internal=False, **kw):
        query = self._sqlite_main_query('sqlite_temp_master', 'view', None, sqlite_include_internal)
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def has_table(self, connection, table_name, schema=None, **kw):
        self._ensure_has_table_connection(connection)
        if schema is not None and schema not in self.get_schema_names(connection, **kw):
            return False
        info = self._get_table_pragma(connection, 'table_info', table_name, schema=schema)
        return bool(info)

    def _get_default_schema_name(self, connection):
        return 'main'

    @reflection.cache
    def get_view_names(self, connection, schema=None, sqlite_include_internal=False, **kw):
        query = self._sqlite_main_query('sqlite_master', 'view', schema, sqlite_include_internal)
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, **kw):
        if schema is not None:
            qschema = self.identifier_preparer.quote_identifier(schema)
            master = f'{qschema}.sqlite_master'
            s = "SELECT sql FROM %s WHERE name = ? AND type='view'" % (master,)
            rs = connection.exec_driver_sql(s, (view_name,))
        else:
            try:
                s = "SELECT sql FROM  (SELECT * FROM sqlite_master UNION ALL   SELECT * FROM sqlite_temp_master) WHERE name = ? AND type='view'"
                rs = connection.exec_driver_sql(s, (view_name,))
            except exc.DBAPIError:
                s = "SELECT sql FROM sqlite_master WHERE name = ? AND type='view'"
                rs = connection.exec_driver_sql(s, (view_name,))
        result = rs.fetchall()
        if result:
            return result[0].sql
        else:
            raise exc.NoSuchTableError(f'{schema}.{view_name}' if schema else view_name)

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        pragma = 'table_info'
        if self.server_version_info >= (3, 31):
            pragma = 'table_xinfo'
        info = self._get_table_pragma(connection, pragma, table_name, schema=schema)
        columns = []
        tablesql = None
        for row in info:
            name = row[1]
            type_ = row[2].upper()
            nullable = not row[3]
            default = row[4]
            primary_key = row[5]
            hidden = row[6] if pragma == 'table_xinfo' else 0
            if hidden == 1:
                continue
            generated = bool(hidden)
            persisted = hidden == 3
            if tablesql is None and generated:
                tablesql = self._get_table_sql(connection, table_name, schema, **kw)
            columns.append(self._get_column_info(name, type_, nullable, default, primary_key, generated, persisted, tablesql))
        if columns:
            return columns
        elif not self.has_table(connection, table_name, schema):
            raise exc.NoSuchTableError(f'{schema}.{table_name}' if schema else table_name)
        else:
            return ReflectionDefaults.columns()

    def _get_column_info(self, name, type_, nullable, default, primary_key, generated, persisted, tablesql):
        if generated:
            type_ = re.sub('generated', '', type_, flags=re.IGNORECASE)
            type_ = re.sub('always', '', type_, flags=re.IGNORECASE).strip()
        coltype = self._resolve_type_affinity(type_)
        if default is not None:
            default = str(default)
        colspec = {'name': name, 'type': coltype, 'nullable': nullable, 'default': default, 'primary_key': primary_key}
        if generated:
            sqltext = ''
            if tablesql:
                pattern = '[^,]*\\s+AS\\s+\\(([^,]*)\\)\\s*(?:virtual|stored)?'
                match = re.search(re.escape(name) + pattern, tablesql, re.IGNORECASE)
                if match:
                    sqltext = match.group(1)
            colspec['computed'] = {'sqltext': sqltext, 'persisted': persisted}
        return colspec

    def _resolve_type_affinity(self, type_):
        """Return a data type from a reflected column, using affinity rules.

        SQLite's goal for universal compatibility introduces some complexity
        during reflection, as a column's defined type might not actually be a
        type that SQLite understands - or indeed, my not be defined *at all*.
        Internally, SQLite handles this with a 'data type affinity' for each
        column definition, mapping to one of 'TEXT', 'NUMERIC', 'INTEGER',
        'REAL', or 'NONE' (raw bits). The algorithm that determines this is
        listed in https://www.sqlite.org/datatype3.html section 2.1.

        This method allows SQLAlchemy to support that algorithm, while still
        providing access to smarter reflection utilities by recognizing
        column definitions that SQLite only supports through affinity (like
        DATE and DOUBLE).

        """
        match = re.match('([\\w ]+)(\\(.*?\\))?', type_)
        if match:
            coltype = match.group(1)
            args = match.group(2)
        else:
            coltype = ''
            args = ''
        if coltype in self.ischema_names:
            coltype = self.ischema_names[coltype]
        elif 'INT' in coltype:
            coltype = sqltypes.INTEGER
        elif 'CHAR' in coltype or 'CLOB' in coltype or 'TEXT' in coltype:
            coltype = sqltypes.TEXT
        elif 'BLOB' in coltype or not coltype:
            coltype = sqltypes.NullType
        elif 'REAL' in coltype or 'FLOA' in coltype or 'DOUB' in coltype:
            coltype = sqltypes.REAL
        else:
            coltype = sqltypes.NUMERIC
        if args is not None:
            args = re.findall('(\\d+)', args)
            try:
                coltype = coltype(*[int(a) for a in args])
            except TypeError:
                util.warn('Could not instantiate type %s with reflected arguments %s; using no arguments.' % (coltype, args))
                coltype = coltype()
        else:
            coltype = coltype()
        return coltype

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        constraint_name = None
        table_data = self._get_table_sql(connection, table_name, schema=schema)
        if table_data:
            PK_PATTERN = 'CONSTRAINT (\\w+) PRIMARY KEY'
            result = re.search(PK_PATTERN, table_data, re.I)
            constraint_name = result.group(1) if result else None
        cols = self.get_columns(connection, table_name, schema, **kw)
        cols = [col for col in cols if col.get('primary_key', 0) > 0]
        cols.sort(key=lambda col: col.get('primary_key'))
        pkeys = [col['name'] for col in cols]
        if pkeys:
            return {'constrained_columns': pkeys, 'name': constraint_name}
        else:
            return ReflectionDefaults.pk_constraint()

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        pragma_fks = self._get_table_pragma(connection, 'foreign_key_list', table_name, schema=schema)
        fks = {}
        for row in pragma_fks:
            numerical_id, rtbl, lcol, rcol = (row[0], row[2], row[3], row[4])
            if not rcol:
                try:
                    referred_pk = self.get_pk_constraint(connection, rtbl, schema=schema, **kw)
                    referred_columns = referred_pk['constrained_columns']
                except exc.NoSuchTableError:
                    referred_columns = []
            else:
                referred_columns = []
            if self._broken_fk_pragma_quotes:
                rtbl = re.sub('^[\\"\\[`\\\']|[\\"\\]`\\\']$', '', rtbl)
            if numerical_id in fks:
                fk = fks[numerical_id]
            else:
                fk = fks[numerical_id] = {'name': None, 'constrained_columns': [], 'referred_schema': schema, 'referred_table': rtbl, 'referred_columns': referred_columns, 'options': {}}
                fks[numerical_id] = fk
            fk['constrained_columns'].append(lcol)
            if rcol:
                fk['referred_columns'].append(rcol)

        def fk_sig(constrained_columns, referred_table, referred_columns):
            return tuple(constrained_columns) + (referred_table,) + tuple(referred_columns)
        keys_by_signature = {fk_sig(fk['constrained_columns'], fk['referred_table'], fk['referred_columns']): fk for fk in fks.values()}
        table_data = self._get_table_sql(connection, table_name, schema=schema)

        def parse_fks():
            if table_data is None:
                return
            FK_PATTERN = '(?:CONSTRAINT (\\w+) +)?FOREIGN KEY *\\( *(.+?) *\\) +REFERENCES +(?:(?:"(.+?)")|([a-z0-9_]+)) *\\( *((?:(?:"[^"]+"|[a-z0-9_]+) *(?:, *)?)+)\\) *((?:ON (?:DELETE|UPDATE) (?:SET NULL|SET DEFAULT|CASCADE|RESTRICT|NO ACTION) *)*)((?:NOT +)?DEFERRABLE)?(?: +INITIALLY +(DEFERRED|IMMEDIATE))?'
            for match in re.finditer(FK_PATTERN, table_data, re.I):
                constraint_name, constrained_columns, referred_quoted_name, referred_name, referred_columns, onupdatedelete, deferrable, initially = match.group(1, 2, 3, 4, 5, 6, 7, 8)
                constrained_columns = list(self._find_cols_in_sig(constrained_columns))
                if not referred_columns:
                    referred_columns = constrained_columns
                else:
                    referred_columns = list(self._find_cols_in_sig(referred_columns))
                referred_name = referred_quoted_name or referred_name
                options = {}
                for token in re.split(' *\\bON\\b *', onupdatedelete.upper()):
                    if token.startswith('DELETE'):
                        ondelete = token[6:].strip()
                        if ondelete and ondelete != 'NO ACTION':
                            options['ondelete'] = ondelete
                    elif token.startswith('UPDATE'):
                        onupdate = token[6:].strip()
                        if onupdate and onupdate != 'NO ACTION':
                            options['onupdate'] = onupdate
                if deferrable:
                    options['deferrable'] = 'NOT' not in deferrable.upper()
                if initially:
                    options['initially'] = initially.upper()
                yield (constraint_name, constrained_columns, referred_name, referred_columns, options)
        fkeys = []
        for constraint_name, constrained_columns, referred_name, referred_columns, options in parse_fks():
            sig = fk_sig(constrained_columns, referred_name, referred_columns)
            if sig not in keys_by_signature:
                util.warn("WARNING: SQL-parsed foreign key constraint '%s' could not be located in PRAGMA foreign_keys for table %s" % (sig, table_name))
                continue
            key = keys_by_signature.pop(sig)
            key['name'] = constraint_name
            key['options'] = options
            fkeys.append(key)
        fkeys.extend(keys_by_signature.values())
        if fkeys:
            return fkeys
        else:
            return ReflectionDefaults.foreign_keys()

    def _find_cols_in_sig(self, sig):
        for match in re.finditer('(?:"(.+?)")|([a-z0-9_]+)', sig, re.I):
            yield (match.group(1) or match.group(2))

    @reflection.cache
    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        auto_index_by_sig = {}
        for idx in self.get_indexes(connection, table_name, schema=schema, include_auto_indexes=True, **kw):
            if not idx['name'].startswith('sqlite_autoindex'):
                continue
            sig = tuple(idx['column_names'])
            auto_index_by_sig[sig] = idx
        table_data = self._get_table_sql(connection, table_name, schema=schema, **kw)
        unique_constraints = []

        def parse_uqs():
            if table_data is None:
                return
            UNIQUE_PATTERN = '(?:CONSTRAINT "?(.+?)"? +)?UNIQUE *\\((.+?)\\)'
            INLINE_UNIQUE_PATTERN = '(?:(".+?")|(?:[\\[`])?([a-z0-9_]+)(?:[\\]`])?) +[a-z0-9_ ]+? +UNIQUE'
            for match in re.finditer(UNIQUE_PATTERN, table_data, re.I):
                name, cols = match.group(1, 2)
                yield (name, list(self._find_cols_in_sig(cols)))
            for match in re.finditer(INLINE_UNIQUE_PATTERN, table_data, re.I):
                cols = list(self._find_cols_in_sig(match.group(1) or match.group(2)))
                yield (None, cols)
        for name, cols in parse_uqs():
            sig = tuple(cols)
            if sig in auto_index_by_sig:
                auto_index_by_sig.pop(sig)
                parsed_constraint = {'name': name, 'column_names': cols}
                unique_constraints.append(parsed_constraint)
        if unique_constraints:
            return unique_constraints
        else:
            return ReflectionDefaults.unique_constraints()

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, **kw):
        table_data = self._get_table_sql(connection, table_name, schema=schema, **kw)
        CHECK_PATTERN = '(?:CONSTRAINT (.+) +)?CHECK *\\( *(.+) *\\),? *'
        cks = []
        for match in re.finditer(CHECK_PATTERN, table_data or '', re.I):
            name = match.group(1)
            if name:
                name = re.sub('^"|"$', '', name)
            cks.append({'sqltext': match.group(2), 'name': name})
        cks.sort(key=lambda d: d['name'] or '~')
        if cks:
            return cks
        else:
            return ReflectionDefaults.check_constraints()

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        pragma_indexes = self._get_table_pragma(connection, 'index_list', table_name, schema=schema)
        indexes = []
        partial_pred_re = re.compile('\\)\\s+where\\s+(.+)', re.IGNORECASE)
        if schema:
            schema_expr = '%s.' % self.identifier_preparer.quote_identifier(schema)
        else:
            schema_expr = ''
        include_auto_indexes = kw.pop('include_auto_indexes', False)
        for row in pragma_indexes:
            if not include_auto_indexes and row[1].startswith('sqlite_autoindex'):
                continue
            indexes.append(dict(name=row[1], column_names=[], unique=row[2], dialect_options={}))
            if len(row) >= 5 and row[4]:
                s = "SELECT sql FROM %(schema)ssqlite_master WHERE name = ? AND type = 'index'" % {'schema': schema_expr}
                rs = connection.exec_driver_sql(s, (row[1],))
                index_sql = rs.scalar()
                predicate_match = partial_pred_re.search(index_sql)
                if predicate_match is None:
                    util.warn('Failed to look up filter predicate of partial index %s' % row[1])
                else:
                    predicate = predicate_match.group(1)
                    indexes[-1]['dialect_options']['sqlite_where'] = text(predicate)
        for idx in list(indexes):
            pragma_index = self._get_table_pragma(connection, 'index_info', idx['name'], schema=schema)
            for row in pragma_index:
                if row[2] is None:
                    util.warn('Skipped unsupported reflection of expression-based index %s' % idx['name'])
                    indexes.remove(idx)
                    break
                else:
                    idx['column_names'].append(row[2])
        indexes.sort(key=lambda d: d['name'] or '~')
        if indexes:
            return indexes
        elif not self.has_table(connection, table_name, schema):
            raise exc.NoSuchTableError(f'{schema}.{table_name}' if schema else table_name)
        else:
            return ReflectionDefaults.indexes()

    def _is_sys_table(self, table_name):
        return table_name in {'sqlite_schema', 'sqlite_master', 'sqlite_temp_schema', 'sqlite_temp_master'}

    @reflection.cache
    def _get_table_sql(self, connection, table_name, schema=None, **kw):
        if schema:
            schema_expr = '%s.' % self.identifier_preparer.quote_identifier(schema)
        else:
            schema_expr = ''
        try:
            s = "SELECT sql FROM  (SELECT * FROM %(schema)ssqlite_master UNION ALL   SELECT * FROM %(schema)ssqlite_temp_master) WHERE name = ? AND type in ('table', 'view')" % {'schema': schema_expr}
            rs = connection.exec_driver_sql(s, (table_name,))
        except exc.DBAPIError:
            s = "SELECT sql FROM %(schema)ssqlite_master WHERE name = ? AND type in ('table', 'view')" % {'schema': schema_expr}
            rs = connection.exec_driver_sql(s, (table_name,))
        value = rs.scalar()
        if value is None and (not self._is_sys_table(table_name)):
            raise exc.NoSuchTableError(f'{schema_expr}{table_name}')
        return value

    def _get_table_pragma(self, connection, pragma, table_name, schema=None):
        quote = self.identifier_preparer.quote_identifier
        if schema is not None:
            statements = [f'PRAGMA {quote(schema)}.']
        else:
            statements = ['PRAGMA main.', 'PRAGMA temp.']
        qtable = quote(table_name)
        for statement in statements:
            statement = f'{statement}{pragma}({qtable})'
            cursor = connection.exec_driver_sql(statement)
            if not cursor._soft_closed:
                result = cursor.fetchall()
            else:
                result = []
            if result:
                return result
        else:
            return []