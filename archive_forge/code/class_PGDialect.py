from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
class PGDialect(default.DefaultDialect):
    name = 'postgresql'
    supports_statement_cache = True
    supports_alter = True
    max_identifier_length = 63
    supports_sane_rowcount = True
    bind_typing = interfaces.BindTyping.RENDER_CASTS
    supports_native_enum = True
    supports_native_boolean = True
    supports_native_uuid = True
    supports_smallserial = True
    supports_sequences = True
    sequences_optional = True
    preexecute_autoincrement_sequences = True
    postfetch_lastrowid = False
    use_insertmanyvalues = True
    returns_native_bytes = True
    insertmanyvalues_implicit_sentinel = InsertmanyvaluesSentinelOpts.ANY_AUTOINCREMENT | InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT | InsertmanyvaluesSentinelOpts.RENDER_SELECT_COL_CASTS
    supports_comments = True
    supports_constraint_comments = True
    supports_default_values = True
    supports_default_metavalue = True
    supports_empty_insert = False
    supports_multivalues_insert = True
    supports_identity_columns = True
    default_paramstyle = 'pyformat'
    ischema_names = ischema_names
    colspecs = colspecs
    statement_compiler = PGCompiler
    ddl_compiler = PGDDLCompiler
    type_compiler_cls = PGTypeCompiler
    preparer = PGIdentifierPreparer
    execution_ctx_cls = PGExecutionContext
    inspector = PGInspector
    update_returning = True
    delete_returning = True
    insert_returning = True
    update_returning_multifrom = True
    delete_returning_multifrom = True
    connection_characteristics = default.DefaultDialect.connection_characteristics
    connection_characteristics = connection_characteristics.union({'postgresql_readonly': PGReadOnlyConnectionCharacteristic(), 'postgresql_deferrable': PGDeferrableConnectionCharacteristic()})
    construct_arguments = [(schema.Index, {'using': False, 'include': None, 'where': None, 'ops': {}, 'concurrently': False, 'with': {}, 'tablespace': None, 'nulls_not_distinct': None}), (schema.Table, {'ignore_search_path': False, 'tablespace': None, 'partition_by': None, 'with_oids': None, 'on_commit': None, 'inherits': None, 'using': None}), (schema.CheckConstraint, {'not_valid': False}), (schema.ForeignKeyConstraint, {'not_valid': False}), (schema.UniqueConstraint, {'nulls_not_distinct': None})]
    reflection_options = ('postgresql_ignore_search_path',)
    _backslash_escapes = True
    _supports_create_index_concurrently = True
    _supports_drop_index_concurrently = True

    def __init__(self, native_inet_types=None, json_serializer=None, json_deserializer=None, **kwargs):
        default.DefaultDialect.__init__(self, **kwargs)
        self._native_inet_types = native_inet_types
        self._json_deserializer = json_deserializer
        self._json_serializer = json_serializer

    def initialize(self, connection):
        super().initialize(connection)
        self.supports_smallserial = self.server_version_info >= (9, 2)
        self._set_backslash_escapes(connection)
        self._supports_drop_index_concurrently = self.server_version_info >= (9, 2)
        self.supports_identity_columns = self.server_version_info >= (10,)

    def get_isolation_level_values(self, dbapi_conn):
        return ('SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ')

    def set_isolation_level(self, dbapi_connection, level):
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL {level}')
        cursor.execute('COMMIT')
        cursor.close()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute('show transaction isolation level')
        val = cursor.fetchone()[0]
        cursor.close()
        return val.upper()

    def set_readonly(self, connection, value):
        raise NotImplementedError()

    def get_readonly(self, connection):
        raise NotImplementedError()

    def set_deferrable(self, connection, value):
        raise NotImplementedError()

    def get_deferrable(self, connection):
        raise NotImplementedError()

    def _split_multihost_from_url(self, url: URL) -> Union[Tuple[None, None], Tuple[Tuple[Optional[str], ...], Tuple[Optional[int], ...]]]:
        hosts: Optional[Tuple[Optional[str], ...]] = None
        ports_str: Union[str, Tuple[Optional[str], ...], None] = None
        integrated_multihost = False
        if 'host' in url.query:
            if isinstance(url.query['host'], (list, tuple)):
                integrated_multihost = True
                hosts, ports_str = zip(*[token.split(':') if ':' in token else (token, None) for token in url.query['host']])
            elif isinstance(url.query['host'], str):
                hosts = tuple(url.query['host'].split(','))
                if 'port' not in url.query and len(hosts) == 1 and (':' in hosts[0]):
                    host_port_match = re.match('^([a-zA-Z0-9\\-\\.]*)(?:\\:(\\d*))?$', hosts[0])
                    if host_port_match:
                        integrated_multihost = True
                        h, p = host_port_match.group(1, 2)
                        if TYPE_CHECKING:
                            assert isinstance(h, str)
                            assert isinstance(p, str)
                        hosts = (h,)
                        ports_str = cast('Tuple[Optional[str], ...]', (p,) if p else (None,))
        if 'port' in url.query:
            if integrated_multihost:
                raise exc.ArgumentError('Can\'t mix \'multihost\' formats together; use "host=h1,h2,h3&port=p1,p2,p3" or "host=h1:p1&host=h2:p2&host=h3:p3" separately')
            if isinstance(url.query['port'], (list, tuple)):
                ports_str = url.query['port']
            elif isinstance(url.query['port'], str):
                ports_str = tuple(url.query['port'].split(','))
        ports: Optional[Tuple[Optional[int], ...]] = None
        if ports_str:
            try:
                ports = tuple((int(x) if x else None for x in ports_str))
            except ValueError:
                raise exc.ArgumentError(f'Received non-integer port arguments: {ports_str}') from None
        if ports and (not hosts and len(ports) > 1 or (hosts and ports and (len(hosts) != len(ports)) and (len(hosts) > 1 or len(ports) > 1))):
            raise exc.ArgumentError("number of hosts and ports don't match")
        if hosts is not None:
            if ports is None:
                ports = tuple((None for _ in hosts))
        return (hosts, ports)

    def do_begin_twophase(self, connection, xid):
        self.do_begin(connection.connection)

    def do_prepare_twophase(self, connection, xid):
        connection.exec_driver_sql("PREPARE TRANSACTION '%s'" % xid)

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        if is_prepared:
            if recover:
                connection.exec_driver_sql('ROLLBACK')
            connection.exec_driver_sql("ROLLBACK PREPARED '%s'" % xid)
            connection.exec_driver_sql('BEGIN')
            self.do_rollback(connection.connection)
        else:
            self.do_rollback(connection.connection)

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if is_prepared:
            if recover:
                connection.exec_driver_sql('ROLLBACK')
            connection.exec_driver_sql("COMMIT PREPARED '%s'" % xid)
            connection.exec_driver_sql('BEGIN')
            self.do_rollback(connection.connection)
        else:
            self.do_commit(connection.connection)

    def do_recover_twophase(self, connection):
        return connection.scalars(sql.text('SELECT gid FROM pg_prepared_xacts')).all()

    def _get_default_schema_name(self, connection):
        return connection.exec_driver_sql('select current_schema()').scalar()

    @reflection.cache
    def has_schema(self, connection, schema, **kw):
        query = select(pg_catalog.pg_namespace.c.nspname).where(pg_catalog.pg_namespace.c.nspname == schema)
        return bool(connection.scalar(query))

    def _pg_class_filter_scope_schema(self, query, schema, scope, pg_class_table=None):
        if pg_class_table is None:
            pg_class_table = pg_catalog.pg_class
        query = query.join(pg_catalog.pg_namespace, pg_catalog.pg_namespace.c.oid == pg_class_table.c.relnamespace)
        if scope is ObjectScope.DEFAULT:
            query = query.where(pg_class_table.c.relpersistence != 't')
        elif scope is ObjectScope.TEMPORARY:
            query = query.where(pg_class_table.c.relpersistence == 't')
        if schema is None:
            query = query.where(pg_catalog.pg_table_is_visible(pg_class_table.c.oid), pg_catalog.pg_namespace.c.nspname != 'pg_catalog')
        else:
            query = query.where(pg_catalog.pg_namespace.c.nspname == schema)
        return query

    def _pg_class_relkind_condition(self, relkinds, pg_class_table=None):
        if pg_class_table is None:
            pg_class_table = pg_catalog.pg_class
        return pg_class_table.c.relkind == sql.any_(_array.array(relkinds))

    @lru_cache()
    def _has_table_query(self, schema):
        query = select(pg_catalog.pg_class.c.relname).where(pg_catalog.pg_class.c.relname == bindparam('table_name'), self._pg_class_relkind_condition(pg_catalog.RELKINDS_ALL_TABLE_LIKE))
        return self._pg_class_filter_scope_schema(query, schema, scope=ObjectScope.ANY)

    @reflection.cache
    def has_table(self, connection, table_name, schema=None, **kw):
        self._ensure_has_table_connection(connection)
        query = self._has_table_query(schema)
        return bool(connection.scalar(query, {'table_name': table_name}))

    @reflection.cache
    def has_sequence(self, connection, sequence_name, schema=None, **kw):
        query = select(pg_catalog.pg_class.c.relname).where(pg_catalog.pg_class.c.relkind == 'S', pg_catalog.pg_class.c.relname == sequence_name)
        query = self._pg_class_filter_scope_schema(query, schema, scope=ObjectScope.ANY)
        return bool(connection.scalar(query))

    @reflection.cache
    def has_type(self, connection, type_name, schema=None, **kw):
        query = select(pg_catalog.pg_type.c.typname).join(pg_catalog.pg_namespace, pg_catalog.pg_namespace.c.oid == pg_catalog.pg_type.c.typnamespace).where(pg_catalog.pg_type.c.typname == type_name)
        if schema is None:
            query = query.where(pg_catalog.pg_type_is_visible(pg_catalog.pg_type.c.oid), pg_catalog.pg_namespace.c.nspname != 'pg_catalog')
        elif schema != '*':
            query = query.where(pg_catalog.pg_namespace.c.nspname == schema)
        return bool(connection.scalar(query))

    def _get_server_version_info(self, connection):
        v = connection.exec_driver_sql('select pg_catalog.version()').scalar()
        m = re.match('.*(?:PostgreSQL|EnterpriseDB) (\\d+)\\.?(\\d+)?(?:\\.(\\d+))?(?:\\.\\d+)?(?:devel|beta)?', v)
        if not m:
            raise AssertionError("Could not determine version from string '%s'" % v)
        return tuple([int(x) for x in m.group(1, 2, 3) if x is not None])

    @reflection.cache
    def get_table_oid(self, connection, table_name, schema=None, **kw):
        """Fetch the oid for schema.table_name."""
        query = select(pg_catalog.pg_class.c.oid).where(pg_catalog.pg_class.c.relname == table_name, self._pg_class_relkind_condition(pg_catalog.RELKINDS_ALL_TABLE_LIKE))
        query = self._pg_class_filter_scope_schema(query, schema, scope=ObjectScope.ANY)
        table_oid = connection.scalar(query)
        if table_oid is None:
            raise exc.NoSuchTableError(f'{schema}.{table_name}' if schema else table_name)
        return table_oid

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        query = select(pg_catalog.pg_namespace.c.nspname).where(pg_catalog.pg_namespace.c.nspname.not_like('pg_%')).order_by(pg_catalog.pg_namespace.c.nspname)
        return connection.scalars(query).all()

    def _get_relnames_for_relkinds(self, connection, schema, relkinds, scope):
        query = select(pg_catalog.pg_class.c.relname).where(self._pg_class_relkind_condition(relkinds))
        query = self._pg_class_filter_scope_schema(query, schema, scope=scope)
        return connection.scalars(query).all()

    @reflection.cache
    def get_table_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, pg_catalog.RELKINDS_TABLE_NO_FOREIGN, scope=ObjectScope.DEFAULT)

    @reflection.cache
    def get_temp_table_names(self, connection, **kw):
        return self._get_relnames_for_relkinds(connection, schema=None, relkinds=pg_catalog.RELKINDS_TABLE_NO_FOREIGN, scope=ObjectScope.TEMPORARY)

    @reflection.cache
    def _get_foreign_table_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, relkinds=('f',), scope=ObjectScope.ANY)

    @reflection.cache
    def get_view_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, pg_catalog.RELKINDS_VIEW, scope=ObjectScope.DEFAULT)

    @reflection.cache
    def get_materialized_view_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, pg_catalog.RELKINDS_MAT_VIEW, scope=ObjectScope.DEFAULT)

    @reflection.cache
    def get_temp_view_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, pg_catalog.RELKINDS_VIEW, scope=ObjectScope.TEMPORARY)

    @reflection.cache
    def get_sequence_names(self, connection, schema=None, **kw):
        return self._get_relnames_for_relkinds(connection, schema, relkinds=('S',), scope=ObjectScope.ANY)

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, **kw):
        query = select(pg_catalog.pg_get_viewdef(pg_catalog.pg_class.c.oid)).select_from(pg_catalog.pg_class).where(pg_catalog.pg_class.c.relname == view_name, self._pg_class_relkind_condition(pg_catalog.RELKINDS_VIEW + pg_catalog.RELKINDS_MAT_VIEW))
        query = self._pg_class_filter_scope_schema(query, schema, scope=ObjectScope.ANY)
        res = connection.scalar(query)
        if res is None:
            raise exc.NoSuchTableError(f'{schema}.{view_name}' if schema else view_name)
        else:
            return res

    def _value_or_raise(self, data, table, schema):
        try:
            return dict(data)[schema, table]
        except KeyError:
            raise exc.NoSuchTableError(f'{schema}.{table}' if schema else table) from None

    def _prepare_filter_names(self, filter_names):
        if filter_names:
            return (True, {'filter_names': filter_names})
        else:
            return (False, {})

    def _kind_to_relkinds(self, kind: ObjectKind) -> Tuple[str, ...]:
        if kind is ObjectKind.ANY:
            return pg_catalog.RELKINDS_ALL_TABLE_LIKE
        relkinds = ()
        if ObjectKind.TABLE in kind:
            relkinds += pg_catalog.RELKINDS_TABLE
        if ObjectKind.VIEW in kind:
            relkinds += pg_catalog.RELKINDS_VIEW
        if ObjectKind.MATERIALIZED_VIEW in kind:
            relkinds += pg_catalog.RELKINDS_MAT_VIEW
        return relkinds

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_columns(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _columns_query(self, schema, has_filter_names, scope, kind):
        generated = pg_catalog.pg_attribute.c.attgenerated.label('generated') if self.server_version_info >= (12,) else sql.null().label('generated')
        if self.server_version_info >= (10,):
            identity = select(sql.func.json_build_object('always', pg_catalog.pg_attribute.c.attidentity == 'a', 'start', pg_catalog.pg_sequence.c.seqstart, 'increment', pg_catalog.pg_sequence.c.seqincrement, 'minvalue', pg_catalog.pg_sequence.c.seqmin, 'maxvalue', pg_catalog.pg_sequence.c.seqmax, 'cache', pg_catalog.pg_sequence.c.seqcache, 'cycle', pg_catalog.pg_sequence.c.seqcycle)).select_from(pg_catalog.pg_sequence).where(pg_catalog.pg_attribute.c.attidentity != '', pg_catalog.pg_sequence.c.seqrelid == sql.cast(sql.cast(pg_catalog.pg_get_serial_sequence(sql.cast(sql.cast(pg_catalog.pg_attribute.c.attrelid, REGCLASS), TEXT), pg_catalog.pg_attribute.c.attname), REGCLASS), OID)).correlate(pg_catalog.pg_attribute).scalar_subquery().label('identity_options')
        else:
            identity = sql.null().label('identity_options')
        default = select(pg_catalog.pg_get_expr(pg_catalog.pg_attrdef.c.adbin, pg_catalog.pg_attrdef.c.adrelid)).select_from(pg_catalog.pg_attrdef).where(pg_catalog.pg_attrdef.c.adrelid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_attrdef.c.adnum == pg_catalog.pg_attribute.c.attnum, pg_catalog.pg_attribute.c.atthasdef).correlate(pg_catalog.pg_attribute).scalar_subquery().label('default')
        relkinds = self._kind_to_relkinds(kind)
        query = select(pg_catalog.pg_attribute.c.attname.label('name'), pg_catalog.format_type(pg_catalog.pg_attribute.c.atttypid, pg_catalog.pg_attribute.c.atttypmod).label('format_type'), default, pg_catalog.pg_attribute.c.attnotnull.label('not_null'), pg_catalog.pg_class.c.relname.label('table_name'), pg_catalog.pg_description.c.description.label('comment'), generated, identity).select_from(pg_catalog.pg_class).outerjoin(pg_catalog.pg_attribute, sql.and_(pg_catalog.pg_class.c.oid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_attribute.c.attnum > 0, ~pg_catalog.pg_attribute.c.attisdropped)).outerjoin(pg_catalog.pg_description, sql.and_(pg_catalog.pg_description.c.objoid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_description.c.objsubid == pg_catalog.pg_attribute.c.attnum)).where(self._pg_class_relkind_condition(relkinds)).order_by(pg_catalog.pg_class.c.relname, pg_catalog.pg_attribute.c.attnum)
        query = self._pg_class_filter_scope_schema(query, schema, scope=scope)
        if has_filter_names:
            query = query.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
        return query

    def get_multi_columns(self, connection, schema, filter_names, scope, kind, **kw):
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._columns_query(schema, has_filter_names, scope, kind)
        rows = connection.execute(query, params).mappings()
        domains = {(d['schema'], d['name']) if not d['visible'] else (d['name'],): d for d in self._load_domains(connection, schema='*', info_cache=kw.get('info_cache'))}
        enums = dict((((rec['name'],), rec) if rec['visible'] else ((rec['schema'], rec['name']), rec) for rec in self._load_enums(connection, schema='*', info_cache=kw.get('info_cache'))))
        columns = self._get_columns_info(rows, domains, enums, schema)
        return columns.items()
    _format_type_args_pattern = re.compile('\\((.*)\\)')
    _format_type_args_delim = re.compile('\\s*,\\s*')
    _format_array_spec_pattern = re.compile('((?:\\[\\])*)$')

    def _reflect_type(self, format_type: Optional[str], domains: dict[str, ReflectedDomain], enums: dict[str, ReflectedEnum], type_description: str) -> sqltypes.TypeEngine[Any]:
        """
        Attempts to reconstruct a column type defined in ischema_names based
        on the information available in the format_type.

        If the `format_type` cannot be associated with a known `ischema_names`,
        it is treated as a reference to a known PostgreSQL named `ENUM` or
        `DOMAIN` type.
        """
        type_description = type_description or 'unknown type'
        if format_type is None:
            util.warn('PostgreSQL format_type() returned NULL for %s' % type_description)
            return sqltypes.NULLTYPE
        attype_args_match = self._format_type_args_pattern.search(format_type)
        if attype_args_match and attype_args_match.group(1):
            attype_args = self._format_type_args_delim.split(attype_args_match.group(1))
        else:
            attype_args = ()
        match_array_dim = self._format_array_spec_pattern.search(format_type)
        array_dim = len(match_array_dim.group(1) or '') // 2
        attype = self._format_type_args_pattern.sub('', format_type)
        attype = self._format_array_spec_pattern.sub('', attype)
        schema_type = self.ischema_names.get(attype.lower(), None)
        args, kwargs = ((), {})
        if attype == 'numeric':
            if len(attype_args) == 2:
                precision, scale = map(int, attype_args)
                args = (precision, scale)
        elif attype == 'double precision':
            args = (53,)
        elif attype == 'integer':
            args = ()
        elif attype in ('timestamp with time zone', 'time with time zone'):
            kwargs['timezone'] = True
            if len(attype_args) == 1:
                kwargs['precision'] = int(attype_args[0])
        elif attype in ('timestamp without time zone', 'time without time zone', 'time'):
            kwargs['timezone'] = False
            if len(attype_args) == 1:
                kwargs['precision'] = int(attype_args[0])
        elif attype == 'bit varying':
            kwargs['varying'] = True
            if len(attype_args) == 1:
                charlen = int(attype_args[0])
                args = (charlen,)
        elif attype.startswith('interval'):
            schema_type = INTERVAL
            field_match = re.match('interval (.+)', attype)
            if field_match:
                kwargs['fields'] = field_match.group(1)
            if len(attype_args) == 1:
                kwargs['precision'] = int(attype_args[0])
        else:
            enum_or_domain_key = tuple(util.quoted_token_parser(attype))
            if enum_or_domain_key in enums:
                schema_type = ENUM
                enum = enums[enum_or_domain_key]
                args = tuple(enum['labels'])
                kwargs['name'] = enum['name']
                if not enum['visible']:
                    kwargs['schema'] = enum['schema']
                args = tuple(enum['labels'])
            elif enum_or_domain_key in domains:
                schema_type = DOMAIN
                domain = domains[enum_or_domain_key]
                data_type = self._reflect_type(domain['type'], domains, enums, type_description="DOMAIN '%s'" % domain['name'])
                args = (domain['name'], data_type)
                kwargs['collation'] = domain['collation']
                kwargs['default'] = domain['default']
                kwargs['not_null'] = not domain['nullable']
                kwargs['create_type'] = False
                if domain['constraints']:
                    check_constraint = domain['constraints'][0]
                    kwargs['constraint_name'] = check_constraint['name']
                    kwargs['check'] = check_constraint['check']
                if not domain['visible']:
                    kwargs['schema'] = domain['schema']
            else:
                try:
                    charlen = int(attype_args[0])
                    args = (charlen, *attype_args[1:])
                except (ValueError, IndexError):
                    args = attype_args
        if not schema_type:
            util.warn("Did not recognize type '%s' of %s" % (attype, type_description))
            return sqltypes.NULLTYPE
        data_type = schema_type(*args, **kwargs)
        if array_dim >= 1:
            data_type = _array.ARRAY(data_type)
        return data_type

    def _get_columns_info(self, rows, domains, enums, schema):
        columns = defaultdict(list)
        for row_dict in rows:
            if row_dict['name'] is None:
                columns[schema, row_dict['table_name']] = ReflectionDefaults.columns()
                continue
            table_cols = columns[schema, row_dict['table_name']]
            coltype = self._reflect_type(row_dict['format_type'], domains, enums, type_description="column '%s'" % row_dict['name'])
            default = row_dict['default']
            name = row_dict['name']
            generated = row_dict['generated']
            nullable = not row_dict['not_null']
            if isinstance(coltype, DOMAIN):
                if not default:
                    if coltype.default is not None:
                        default = coltype.default
                nullable = nullable and (not coltype.not_null)
            identity = row_dict['identity_options']
            if generated not in (None, '', b'\x00'):
                computed = dict(sqltext=default, persisted=generated in ('s', b's'))
                default = None
            else:
                computed = None
            autoincrement = False
            if default is not None:
                match = re.search("(nextval\\(')([^']+)('.*$)", default)
                if match is not None:
                    if issubclass(coltype._type_affinity, sqltypes.Integer):
                        autoincrement = True
                    if '.' not in match.group(2) and schema is not None:
                        default = match.group(1) + '"%s"' % schema + '.' + match.group(2) + match.group(3)
            column_info = {'name': name, 'type': coltype, 'nullable': nullable, 'default': default, 'autoincrement': autoincrement or identity is not None, 'comment': row_dict['comment']}
            if computed is not None:
                column_info['computed'] = computed
            if identity is not None:
                column_info['identity'] = identity
            table_cols.append(column_info)
        return columns

    @lru_cache()
    def _table_oids_query(self, schema, has_filter_names, scope, kind):
        relkinds = self._kind_to_relkinds(kind)
        oid_q = select(pg_catalog.pg_class.c.oid, pg_catalog.pg_class.c.relname).where(self._pg_class_relkind_condition(relkinds))
        oid_q = self._pg_class_filter_scope_schema(oid_q, schema, scope=scope)
        if has_filter_names:
            oid_q = oid_q.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
        return oid_q

    @reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('filter_names', InternalTraversal.dp_string_list), ('kind', InternalTraversal.dp_plain_obj), ('scope', InternalTraversal.dp_plain_obj))
    def _get_table_oids(self, connection, schema, filter_names, scope, kind, **kw):
        has_filter_names, params = self._prepare_filter_names(filter_names)
        oid_q = self._table_oids_query(schema, has_filter_names, scope, kind)
        result = connection.execute(oid_q, params)
        return result.all()

    @lru_cache()
    def _constraint_query(self, is_unique):
        con_sq = select(pg_catalog.pg_constraint.c.conrelid, pg_catalog.pg_constraint.c.conname, pg_catalog.pg_constraint.c.conindid, sql.func.unnest(pg_catalog.pg_constraint.c.conkey).label('attnum'), sql.func.generate_subscripts(pg_catalog.pg_constraint.c.conkey, 1).label('ord'), pg_catalog.pg_description.c.description).outerjoin(pg_catalog.pg_description, pg_catalog.pg_description.c.objoid == pg_catalog.pg_constraint.c.oid).where(pg_catalog.pg_constraint.c.contype == bindparam('contype'), pg_catalog.pg_constraint.c.conrelid.in_(bindparam('oids'))).subquery('con')
        attr_sq = select(con_sq.c.conrelid, con_sq.c.conname, con_sq.c.conindid, con_sq.c.description, con_sq.c.ord, pg_catalog.pg_attribute.c.attname).select_from(pg_catalog.pg_attribute).join(con_sq, sql.and_(pg_catalog.pg_attribute.c.attnum == con_sq.c.attnum, pg_catalog.pg_attribute.c.attrelid == con_sq.c.conrelid)).where(con_sq.c.conrelid.in_(bindparam('oids'))).subquery('attr')
        constraint_query = select(attr_sq.c.conrelid, sql.func.array_agg(aggregate_order_by(attr_sq.c.attname.cast(TEXT), attr_sq.c.ord)).label('cols'), attr_sq.c.conname, sql.func.min(attr_sq.c.description).label('description')).group_by(attr_sq.c.conrelid, attr_sq.c.conname).order_by(attr_sq.c.conrelid, attr_sq.c.conname)
        if is_unique:
            if self.server_version_info >= (15,):
                constraint_query = constraint_query.join(pg_catalog.pg_index, attr_sq.c.conindid == pg_catalog.pg_index.c.indexrelid).add_columns(sql.func.bool_and(pg_catalog.pg_index.c.indnullsnotdistinct).label('indnullsnotdistinct'))
            else:
                constraint_query = constraint_query.add_columns(sql.false().label('indnullsnotdistinct'))
        else:
            constraint_query = constraint_query.add_columns(sql.null().label('extra'))
        return constraint_query

    def _reflect_constraint(self, connection, contype, schema, filter_names, scope, kind, **kw):
        table_oids = self._get_table_oids(connection, schema, filter_names, scope, kind, **kw)
        batches = list(table_oids)
        is_unique = contype == 'u'
        while batches:
            batch = batches[0:3000]
            batches[0:3000] = []
            result = connection.execute(self._constraint_query(is_unique), {'oids': [r[0] for r in batch], 'contype': contype})
            result_by_oid = defaultdict(list)
            for oid, cols, constraint_name, comment, extra in result:
                result_by_oid[oid].append((cols, constraint_name, comment, extra))
            for oid, tablename in batch:
                for_oid = result_by_oid.get(oid, ())
                if for_oid:
                    for cols, constraint, comment, extra in for_oid:
                        if is_unique:
                            yield (tablename, cols, constraint, comment, {'nullsnotdistinct': extra})
                        else:
                            yield (tablename, cols, constraint, comment, None)
                else:
                    yield (tablename, None, None, None, None)

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_pk_constraint(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    def get_multi_pk_constraint(self, connection, schema, filter_names, scope, kind, **kw):
        result = self._reflect_constraint(connection, 'p', schema, filter_names, scope, kind, **kw)
        default = ReflectionDefaults.pk_constraint
        return (((schema, table_name), {'constrained_columns': [] if cols is None else cols, 'name': pk_name, 'comment': comment} if pk_name is not None else default()) for table_name, cols, pk_name, comment, _ in result)

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, postgresql_ignore_search_path=False, **kw):
        data = self.get_multi_foreign_keys(connection, schema=schema, filter_names=[table_name], postgresql_ignore_search_path=postgresql_ignore_search_path, scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _foreing_key_query(self, schema, has_filter_names, scope, kind):
        pg_class_ref = pg_catalog.pg_class.alias('cls_ref')
        pg_namespace_ref = pg_catalog.pg_namespace.alias('nsp_ref')
        relkinds = self._kind_to_relkinds(kind)
        query = select(pg_catalog.pg_class.c.relname, pg_catalog.pg_constraint.c.conname, sql.case((pg_catalog.pg_constraint.c.oid.is_not(None), pg_catalog.pg_get_constraintdef(pg_catalog.pg_constraint.c.oid, True)), else_=None), pg_namespace_ref.c.nspname, pg_catalog.pg_description.c.description).select_from(pg_catalog.pg_class).outerjoin(pg_catalog.pg_constraint, sql.and_(pg_catalog.pg_class.c.oid == pg_catalog.pg_constraint.c.conrelid, pg_catalog.pg_constraint.c.contype == 'f')).outerjoin(pg_class_ref, pg_class_ref.c.oid == pg_catalog.pg_constraint.c.confrelid).outerjoin(pg_namespace_ref, pg_class_ref.c.relnamespace == pg_namespace_ref.c.oid).outerjoin(pg_catalog.pg_description, pg_catalog.pg_description.c.objoid == pg_catalog.pg_constraint.c.oid).order_by(pg_catalog.pg_class.c.relname, pg_catalog.pg_constraint.c.conname).where(self._pg_class_relkind_condition(relkinds))
        query = self._pg_class_filter_scope_schema(query, schema, scope)
        if has_filter_names:
            query = query.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
        return query

    @util.memoized_property
    def _fk_regex_pattern(self):
        qtoken = '(?:"[^"]+"|[A-Za-z0-9_]+?)'
        return re.compile(f'FOREIGN KEY \\((.*?)\\) REFERENCES (?:({qtoken})\\.)?({qtoken})\\(((?:{qtoken}(?: *, *)?)+)\\)[\\s]?(MATCH (FULL|PARTIAL|SIMPLE)+)?[\\s]?(ON UPDATE (CASCADE|RESTRICT|NO ACTION|SET NULL|SET DEFAULT)+)?[\\s]?(ON DELETE (CASCADE|RESTRICT|NO ACTION|SET NULL|SET DEFAULT)+)?[\\s]?(DEFERRABLE|NOT DEFERRABLE)?[\\s]?(INITIALLY (DEFERRED|IMMEDIATE)+)?')

    def get_multi_foreign_keys(self, connection, schema, filter_names, scope, kind, postgresql_ignore_search_path=False, **kw):
        preparer = self.identifier_preparer
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._foreing_key_query(schema, has_filter_names, scope, kind)
        result = connection.execute(query, params)
        FK_REGEX = self._fk_regex_pattern
        fkeys = defaultdict(list)
        default = ReflectionDefaults.foreign_keys
        for table_name, conname, condef, conschema, comment in result:
            if conname is None:
                fkeys[schema, table_name] = default()
                continue
            table_fks = fkeys[schema, table_name]
            m = re.search(FK_REGEX, condef).groups()
            constrained_columns, referred_schema, referred_table, referred_columns, _, match, _, onupdate, _, ondelete, deferrable, _, initially = m
            if deferrable is not None:
                deferrable = True if deferrable == 'DEFERRABLE' else False
            constrained_columns = [preparer._unquote_identifier(x) for x in re.split('\\s*,\\s*', constrained_columns)]
            if postgresql_ignore_search_path:
                if conschema != self.default_schema_name:
                    referred_schema = conschema
                else:
                    referred_schema = schema
            elif referred_schema:
                referred_schema = preparer._unquote_identifier(referred_schema)
            elif schema is not None and schema == conschema:
                referred_schema = schema
            referred_table = preparer._unquote_identifier(referred_table)
            referred_columns = [preparer._unquote_identifier(x) for x in re.split('\\s*,\\s', referred_columns)]
            options = {k: v for k, v in [('onupdate', onupdate), ('ondelete', ondelete), ('initially', initially), ('deferrable', deferrable), ('match', match)] if v is not None and v != 'NO ACTION'}
            fkey_d = {'name': conname, 'constrained_columns': constrained_columns, 'referred_schema': referred_schema, 'referred_table': referred_table, 'referred_columns': referred_columns, 'options': options, 'comment': comment}
            table_fks.append(fkey_d)
        return fkeys.items()

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_indexes(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @util.memoized_property
    def _index_query(self):
        pg_class_index = pg_catalog.pg_class.alias('cls_idx')
        idx_sq = select(pg_catalog.pg_index.c.indexrelid, pg_catalog.pg_index.c.indrelid, sql.func.unnest(pg_catalog.pg_index.c.indkey).label('attnum'), sql.func.generate_subscripts(pg_catalog.pg_index.c.indkey, 1).label('ord')).where(~pg_catalog.pg_index.c.indisprimary, pg_catalog.pg_index.c.indrelid.in_(bindparam('oids'))).subquery('idx')
        attr_sq = select(idx_sq.c.indexrelid, idx_sq.c.indrelid, idx_sq.c.ord, sql.case((idx_sq.c.attnum == 0, pg_catalog.pg_get_indexdef(idx_sq.c.indexrelid, idx_sq.c.ord + 1, True)), else_=pg_catalog.pg_attribute.c.attname.cast(TEXT)).label('element'), (idx_sq.c.attnum == 0).label('is_expr')).select_from(idx_sq).outerjoin(pg_catalog.pg_attribute, sql.and_(pg_catalog.pg_attribute.c.attnum == idx_sq.c.attnum, pg_catalog.pg_attribute.c.attrelid == idx_sq.c.indrelid)).where(idx_sq.c.indrelid.in_(bindparam('oids'))).subquery('idx_attr')
        cols_sq = select(attr_sq.c.indexrelid, sql.func.min(attr_sq.c.indrelid), sql.func.array_agg(aggregate_order_by(attr_sq.c.element, attr_sq.c.ord)).label('elements'), sql.func.array_agg(aggregate_order_by(attr_sq.c.is_expr, attr_sq.c.ord)).label('elements_is_expr')).group_by(attr_sq.c.indexrelid).subquery('idx_cols')
        if self.server_version_info >= (11, 0):
            indnkeyatts = pg_catalog.pg_index.c.indnkeyatts
        else:
            indnkeyatts = sql.null().label('indnkeyatts')
        if self.server_version_info >= (15,):
            nulls_not_distinct = pg_catalog.pg_index.c.indnullsnotdistinct
        else:
            nulls_not_distinct = sql.false().label('indnullsnotdistinct')
        return select(pg_catalog.pg_index.c.indrelid, pg_class_index.c.relname.label('relname_index'), pg_catalog.pg_index.c.indisunique, pg_catalog.pg_constraint.c.conrelid.is_not(None).label('has_constraint'), pg_catalog.pg_index.c.indoption, pg_class_index.c.reloptions, pg_catalog.pg_am.c.amname, sql.case((pg_catalog.pg_index.c.indpred.is_not(None), pg_catalog.pg_get_expr(pg_catalog.pg_index.c.indpred, pg_catalog.pg_index.c.indrelid)), else_=None).label('filter_definition'), indnkeyatts, nulls_not_distinct, cols_sq.c.elements, cols_sq.c.elements_is_expr).select_from(pg_catalog.pg_index).where(pg_catalog.pg_index.c.indrelid.in_(bindparam('oids')), ~pg_catalog.pg_index.c.indisprimary).join(pg_class_index, pg_catalog.pg_index.c.indexrelid == pg_class_index.c.oid).join(pg_catalog.pg_am, pg_class_index.c.relam == pg_catalog.pg_am.c.oid).outerjoin(cols_sq, pg_catalog.pg_index.c.indexrelid == cols_sq.c.indexrelid).outerjoin(pg_catalog.pg_constraint, sql.and_(pg_catalog.pg_index.c.indrelid == pg_catalog.pg_constraint.c.conrelid, pg_catalog.pg_index.c.indexrelid == pg_catalog.pg_constraint.c.conindid, pg_catalog.pg_constraint.c.contype == sql.any_(_array.array(('p', 'u', 'x'))))).order_by(pg_catalog.pg_index.c.indrelid, pg_class_index.c.relname)

    def get_multi_indexes(self, connection, schema, filter_names, scope, kind, **kw):
        table_oids = self._get_table_oids(connection, schema, filter_names, scope, kind, **kw)
        indexes = defaultdict(list)
        default = ReflectionDefaults.indexes
        batches = list(table_oids)
        while batches:
            batch = batches[0:3000]
            batches[0:3000] = []
            result = connection.execute(self._index_query, {'oids': [r[0] for r in batch]}).mappings()
            result_by_oid = defaultdict(list)
            for row_dict in result:
                result_by_oid[row_dict['indrelid']].append(row_dict)
            for oid, table_name in batch:
                if oid not in result_by_oid:
                    indexes[schema, table_name] = default()
                    continue
                for row in result_by_oid[oid]:
                    index_name = row['relname_index']
                    table_indexes = indexes[schema, table_name]
                    all_elements = row['elements']
                    all_elements_is_expr = row['elements_is_expr']
                    indnkeyatts = row['indnkeyatts']
                    if indnkeyatts and len(all_elements) > indnkeyatts:
                        inc_cols = all_elements[indnkeyatts:]
                        idx_elements = all_elements[:indnkeyatts]
                        idx_elements_is_expr = all_elements_is_expr[:indnkeyatts]
                        assert all((not is_expr for is_expr in all_elements_is_expr[indnkeyatts:]))
                    else:
                        idx_elements = all_elements
                        idx_elements_is_expr = all_elements_is_expr
                        inc_cols = []
                    index = {'name': index_name, 'unique': row['indisunique']}
                    if any(idx_elements_is_expr):
                        index['column_names'] = [None if is_expr else expr for expr, is_expr in zip(idx_elements, idx_elements_is_expr)]
                        index['expressions'] = idx_elements
                    else:
                        index['column_names'] = idx_elements
                    sorting = {}
                    for col_index, col_flags in enumerate(row['indoption']):
                        col_sorting = ()
                        if col_flags & 1:
                            col_sorting += ('desc',)
                            if not col_flags & 2:
                                col_sorting += ('nulls_last',)
                        elif col_flags & 2:
                            col_sorting += ('nulls_first',)
                        if col_sorting:
                            sorting[idx_elements[col_index]] = col_sorting
                    if sorting:
                        index['column_sorting'] = sorting
                    if row['has_constraint']:
                        index['duplicates_constraint'] = index_name
                    dialect_options = {}
                    if row['reloptions']:
                        dialect_options['postgresql_with'] = dict([option.split('=') for option in row['reloptions']])
                    amname = row['amname']
                    if amname != 'btree':
                        dialect_options['postgresql_using'] = row['amname']
                    if row['filter_definition']:
                        dialect_options['postgresql_where'] = row['filter_definition']
                    if self.server_version_info >= (11,):
                        index['include_columns'] = inc_cols
                        dialect_options['postgresql_include'] = inc_cols
                    if row['indnullsnotdistinct']:
                        dialect_options['postgresql_nulls_not_distinct'] = row['indnullsnotdistinct']
                    if dialect_options:
                        index['dialect_options'] = dialect_options
                    table_indexes.append(index)
        return indexes.items()

    @reflection.cache
    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_unique_constraints(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    def get_multi_unique_constraints(self, connection, schema, filter_names, scope, kind, **kw):
        result = self._reflect_constraint(connection, 'u', schema, filter_names, scope, kind, **kw)
        uniques = defaultdict(list)
        default = ReflectionDefaults.unique_constraints
        for table_name, cols, con_name, comment, options in result:
            if con_name is None:
                uniques[schema, table_name] = default()
                continue
            uc_dict = {'column_names': cols, 'name': con_name, 'comment': comment}
            if options:
                if options['nullsnotdistinct']:
                    uc_dict['dialect_options'] = {'postgresql_nulls_not_distinct': options['nullsnotdistinct']}
            uniques[schema, table_name].append(uc_dict)
        return uniques.items()

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_table_comment(connection, schema, [table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _comment_query(self, schema, has_filter_names, scope, kind):
        relkinds = self._kind_to_relkinds(kind)
        query = select(pg_catalog.pg_class.c.relname, pg_catalog.pg_description.c.description).select_from(pg_catalog.pg_class).outerjoin(pg_catalog.pg_description, sql.and_(pg_catalog.pg_class.c.oid == pg_catalog.pg_description.c.objoid, pg_catalog.pg_description.c.objsubid == 0)).where(self._pg_class_relkind_condition(relkinds))
        query = self._pg_class_filter_scope_schema(query, schema, scope)
        if has_filter_names:
            query = query.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
        return query

    def get_multi_table_comment(self, connection, schema, filter_names, scope, kind, **kw):
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._comment_query(schema, has_filter_names, scope, kind)
        result = connection.execute(query, params)
        default = ReflectionDefaults.table_comment
        return (((schema, table), {'text': comment} if comment is not None else default()) for table, comment in result)

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, **kw):
        data = self.get_multi_check_constraints(connection, schema, [table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _check_constraint_query(self, schema, has_filter_names, scope, kind):
        relkinds = self._kind_to_relkinds(kind)
        query = select(pg_catalog.pg_class.c.relname, pg_catalog.pg_constraint.c.conname, sql.case((pg_catalog.pg_constraint.c.oid.is_not(None), pg_catalog.pg_get_constraintdef(pg_catalog.pg_constraint.c.oid, True)), else_=None), pg_catalog.pg_description.c.description).select_from(pg_catalog.pg_class).outerjoin(pg_catalog.pg_constraint, sql.and_(pg_catalog.pg_class.c.oid == pg_catalog.pg_constraint.c.conrelid, pg_catalog.pg_constraint.c.contype == 'c')).outerjoin(pg_catalog.pg_description, pg_catalog.pg_description.c.objoid == pg_catalog.pg_constraint.c.oid).order_by(pg_catalog.pg_class.c.relname, pg_catalog.pg_constraint.c.conname).where(self._pg_class_relkind_condition(relkinds))
        query = self._pg_class_filter_scope_schema(query, schema, scope)
        if has_filter_names:
            query = query.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
        return query

    def get_multi_check_constraints(self, connection, schema, filter_names, scope, kind, **kw):
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._check_constraint_query(schema, has_filter_names, scope, kind)
        result = connection.execute(query, params)
        check_constraints = defaultdict(list)
        default = ReflectionDefaults.check_constraints
        for table_name, check_name, src, comment in result:
            if check_name is None and src is None:
                check_constraints[schema, table_name] = default()
                continue
            m = re.match('^CHECK *\\((.+)\\)( NO INHERIT)?( NOT VALID)?$', src, flags=re.DOTALL)
            if not m:
                util.warn('Could not parse CHECK constraint text: %r' % src)
                sqltext = ''
            else:
                sqltext = re.compile('^[\\s\\n]*\\((.+)\\)[\\s\\n]*$', flags=re.DOTALL).sub('\\1', m.group(1))
            entry = {'name': check_name, 'sqltext': sqltext, 'comment': comment}
            if m:
                do = {}
                if ' NOT VALID' in m.groups():
                    do['not_valid'] = True
                if ' NO INHERIT' in m.groups():
                    do['no_inherit'] = True
                if do:
                    entry['dialect_options'] = do
            check_constraints[schema, table_name].append(entry)
        return check_constraints.items()

    def _pg_type_filter_schema(self, query, schema):
        if schema is None:
            query = query.where(pg_catalog.pg_type_is_visible(pg_catalog.pg_type.c.oid), pg_catalog.pg_namespace.c.nspname != 'pg_catalog')
        elif schema != '*':
            query = query.where(pg_catalog.pg_namespace.c.nspname == schema)
        return query

    @lru_cache()
    def _enum_query(self, schema):
        lbl_agg_sq = select(pg_catalog.pg_enum.c.enumtypid, sql.func.array_agg(aggregate_order_by(pg_catalog.pg_enum.c.enumlabel.cast(TEXT), pg_catalog.pg_enum.c.enumsortorder)).label('labels')).group_by(pg_catalog.pg_enum.c.enumtypid).subquery('lbl_agg')
        query = select(pg_catalog.pg_type.c.typname.label('name'), pg_catalog.pg_type_is_visible(pg_catalog.pg_type.c.oid).label('visible'), pg_catalog.pg_namespace.c.nspname.label('schema'), lbl_agg_sq.c.labels.label('labels')).join(pg_catalog.pg_namespace, pg_catalog.pg_namespace.c.oid == pg_catalog.pg_type.c.typnamespace).outerjoin(lbl_agg_sq, pg_catalog.pg_type.c.oid == lbl_agg_sq.c.enumtypid).where(pg_catalog.pg_type.c.typtype == 'e').order_by(pg_catalog.pg_namespace.c.nspname, pg_catalog.pg_type.c.typname)
        return self._pg_type_filter_schema(query, schema)

    @reflection.cache
    def _load_enums(self, connection, schema=None, **kw):
        if not self.supports_native_enum:
            return []
        result = connection.execute(self._enum_query(schema))
        enums = []
        for name, visible, schema, labels in result:
            enums.append({'name': name, 'schema': schema, 'visible': visible, 'labels': [] if labels is None else labels})
        return enums

    @lru_cache()
    def _domain_query(self, schema):
        con_sq = select(pg_catalog.pg_constraint.c.contypid, sql.func.array_agg(pg_catalog.pg_get_constraintdef(pg_catalog.pg_constraint.c.oid, True)).label('condefs'), sql.func.array_agg(pg_catalog.pg_constraint.c.conname.cast(TEXT)).label('connames')).where(pg_catalog.pg_constraint.c.contypid != 0).group_by(pg_catalog.pg_constraint.c.contypid).subquery('domain_constraints')
        query = select(pg_catalog.pg_type.c.typname.label('name'), pg_catalog.format_type(pg_catalog.pg_type.c.typbasetype, pg_catalog.pg_type.c.typtypmod).label('attype'), (~pg_catalog.pg_type.c.typnotnull).label('nullable'), pg_catalog.pg_type.c.typdefault.label('default'), pg_catalog.pg_type_is_visible(pg_catalog.pg_type.c.oid).label('visible'), pg_catalog.pg_namespace.c.nspname.label('schema'), con_sq.c.condefs, con_sq.c.connames, pg_catalog.pg_collation.c.collname).join(pg_catalog.pg_namespace, pg_catalog.pg_namespace.c.oid == pg_catalog.pg_type.c.typnamespace).outerjoin(pg_catalog.pg_collation, pg_catalog.pg_type.c.typcollation == pg_catalog.pg_collation.c.oid).outerjoin(con_sq, pg_catalog.pg_type.c.oid == con_sq.c.contypid).where(pg_catalog.pg_type.c.typtype == 'd').order_by(pg_catalog.pg_namespace.c.nspname, pg_catalog.pg_type.c.typname)
        return self._pg_type_filter_schema(query, schema)

    @reflection.cache
    def _load_domains(self, connection, schema=None, **kw):
        result = connection.execute(self._domain_query(schema))
        domains: List[ReflectedDomain] = []
        for domain in result.mappings():
            attype = re.search('([^\\(]+)', domain['attype']).group(1)
            constraints: List[ReflectedDomainConstraint] = []
            if domain['connames']:
                sorted_constraints = sorted(zip(domain['connames'], domain['condefs']), key=lambda t: t[0])
                for name, def_ in sorted_constraints:
                    check = def_[7:-1]
                    constraints.append({'name': name, 'check': check})
            domain_rec: ReflectedDomain = {'name': domain['name'], 'schema': domain['schema'], 'visible': domain['visible'], 'type': attype, 'nullable': domain['nullable'], 'default': domain['default'], 'constraints': constraints, 'collation': domain['collname']}
            domains.append(domain_rec)
        return domains

    def _set_backslash_escapes(self, connection):
        std_string = connection.exec_driver_sql('show standard_conforming_strings').scalar()
        self._backslash_escapes = std_string == 'off'