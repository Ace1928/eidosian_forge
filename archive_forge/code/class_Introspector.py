from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
class Introspector(object):
    pk_classes = [AutoField, IntegerField]

    def __init__(self, metadata, schema=None):
        self.metadata = metadata
        self.schema = schema

    def __repr__(self):
        return '<Introspector: %s>' % self.metadata.database

    @classmethod
    def from_database(cls, database, schema=None):
        if isinstance(database, Proxy):
            if database.obj is None:
                raise ValueError('Cannot introspect an uninitialized Proxy.')
            database = database.obj
        if CockroachDatabase and isinstance(database, CockroachDatabase):
            metadata = CockroachDBMetadata(database)
        elif isinstance(database, PostgresqlDatabase):
            metadata = PostgresqlMetadata(database)
        elif isinstance(database, MySQLDatabase):
            metadata = MySQLMetadata(database)
        elif isinstance(database, SqliteDatabase):
            metadata = SqliteMetadata(database)
        else:
            raise ValueError('Introspection not supported for %r' % database)
        return cls(metadata, schema=schema)

    def get_database_class(self):
        return type(self.metadata.database)

    def get_database_name(self):
        return self.metadata.database.database

    def get_database_kwargs(self):
        return self.metadata.database.connect_params

    def get_additional_imports(self):
        if self.metadata.requires_extension:
            return '\n' + self.metadata.extension_import
        return ''

    def make_model_name(self, table, snake_case=True):
        if snake_case:
            table = make_snake_case(table)
        model = re.sub('[^\\w]+', '', table)
        model_name = ''.join((sub.title() for sub in model.split('_')))
        if not model_name[0].isalpha():
            model_name = 'T' + model_name
        return model_name

    def make_column_name(self, column, is_foreign_key=False, snake_case=True):
        column = column.strip()
        if snake_case:
            column = make_snake_case(column)
        column = column.lower()
        if is_foreign_key:
            column = re.sub('_id$', '', column) or column
        column = re.sub('[^\\w]+', '_', column)
        if column in RESERVED_WORDS:
            column += '_'
        if len(column) and column[0].isdigit():
            column = '_' + column
        return column

    def introspect(self, table_names=None, literal_column_names=False, include_views=False, snake_case=True):
        tables = self.metadata.database.get_tables(schema=self.schema)
        if include_views:
            views = self.metadata.database.get_views(schema=self.schema)
            tables.extend([view.name for view in views])
        if table_names is not None:
            tables = [table for table in tables if table in table_names]
        table_set = set(tables)
        columns = {}
        primary_keys = {}
        foreign_keys = {}
        model_names = {}
        indexes = {}
        for table in tables:
            table_indexes = self.metadata.get_indexes(table, self.schema)
            table_columns = self.metadata.get_columns(table, self.schema)
            try:
                foreign_keys[table] = self.metadata.get_foreign_keys(table, self.schema)
            except ValueError as exc:
                err(*exc.args)
                foreign_keys[table] = []
            else:
                if table_names is not None:
                    for foreign_key in foreign_keys[table]:
                        if foreign_key.dest_table not in table_set:
                            tables.append(foreign_key.dest_table)
                            table_set.add(foreign_key.dest_table)
            model_names[table] = self.make_model_name(table, snake_case)
            lower_col_names = set((column_name.lower() for column_name in table_columns))
            fks = set((fk_col.column for fk_col in foreign_keys[table]))
            for col_name, column in table_columns.items():
                if literal_column_names:
                    new_name = re.sub('[^\\w]+', '_', col_name)
                else:
                    new_name = self.make_column_name(col_name, col_name in fks, snake_case)
                lower_name = col_name.lower()
                if lower_name.endswith('_id') and new_name in lower_col_names:
                    new_name = col_name.lower()
                column.name = new_name
            for index in table_indexes:
                if len(index.columns) == 1:
                    column = index.columns[0]
                    if column in table_columns:
                        table_columns[column].unique = index.unique
                        table_columns[column].index = True
            primary_keys[table] = self.metadata.get_primary_keys(table, self.schema)
            columns[table] = table_columns
            indexes[table] = table_indexes
        related_names = {}
        sort_fn = lambda foreign_key: foreign_key.column
        for table in tables:
            models_referenced = set()
            for foreign_key in sorted(foreign_keys[table], key=sort_fn):
                try:
                    column = columns[table][foreign_key.column]
                except KeyError:
                    continue
                dest_table = foreign_key.dest_table
                if dest_table in models_referenced:
                    related_names[column] = '%s_%s_set' % (dest_table, column.name)
                else:
                    models_referenced.add(dest_table)
        for table in tables:
            for foreign_key in foreign_keys[table]:
                src = columns[foreign_key.table][foreign_key.column]
                try:
                    dest = columns[foreign_key.dest_table][foreign_key.dest_column]
                except KeyError:
                    dest = None
                src.set_foreign_key(foreign_key=foreign_key, model_names=model_names, dest=dest, related_name=related_names.get(src))
        return DatabaseMetadata(columns, primary_keys, foreign_keys, model_names, indexes)

    def generate_models(self, skip_invalid=False, table_names=None, literal_column_names=False, bare_fields=False, include_views=False):
        database = self.introspect(table_names, literal_column_names, include_views)
        models = {}

        class BaseModel(Model):

            class Meta:
                database = self.metadata.database
                schema = self.schema
        pending = set()

        def _create_model(table, models):
            pending.add(table)
            for foreign_key in database.foreign_keys[table]:
                dest = foreign_key.dest_table
                if dest not in models and dest != table:
                    if dest in pending:
                        warnings.warn('Possible reference cycle found between %s and %s' % (table, dest))
                    else:
                        _create_model(dest, models)
            primary_keys = []
            columns = database.columns[table]
            for column_name, column in columns.items():
                if column.primary_key:
                    primary_keys.append(column.name)
            multi_column_indexes = database.multi_column_indexes(table)
            column_indexes = database.column_indexes(table)

            class Meta:
                indexes = multi_column_indexes
                table_name = table
            composite_key = False
            if len(primary_keys) == 0:
                if 'id' not in columns:
                    Meta.primary_key = False
                else:
                    primary_keys = columns.keys()
            if len(primary_keys) > 1:
                Meta.primary_key = CompositeKey(*[field.name for col, field in columns.items() if col in primary_keys])
                composite_key = True
            attrs = {'Meta': Meta}
            for column_name, column in columns.items():
                FieldClass = column.field_class
                if FieldClass is not ForeignKeyField and bare_fields:
                    FieldClass = BareField
                elif FieldClass is UnknownField:
                    FieldClass = BareField
                params = {'column_name': column_name, 'null': column.nullable}
                if column.primary_key and composite_key:
                    if FieldClass is AutoField:
                        FieldClass = IntegerField
                    params['primary_key'] = False
                elif column.primary_key and FieldClass is not AutoField:
                    params['primary_key'] = True
                if column.is_foreign_key():
                    if column.is_self_referential_fk():
                        params['model'] = 'self'
                    else:
                        dest_table = column.foreign_key.dest_table
                        if dest_table in models:
                            params['model'] = models[dest_table]
                        else:
                            FieldClass = DeferredForeignKey
                            params['rel_model_name'] = dest_table
                    if column.to_field:
                        params['field'] = column.to_field
                    params['backref'] = '%s_%s_rel' % (table, column_name)
                if column.default is not None:
                    constraint = SQL('DEFAULT %s' % column.default)
                    params['constraints'] = [constraint]
                if not column.is_primary_key():
                    if column_name in column_indexes:
                        if column_indexes[column_name]:
                            params['unique'] = True
                        elif not column.is_foreign_key():
                            params['index'] = True
                    else:
                        params['index'] = False
                attrs[column.name] = FieldClass(**params)
            try:
                models[table] = type(str(table), (BaseModel,), attrs)
            except ValueError:
                if not skip_invalid:
                    raise
            finally:
                if table in pending:
                    pending.remove(table)
        for table, model in sorted(database.model_names.items()):
            if table not in models:
                _create_model(table, models)
        return models