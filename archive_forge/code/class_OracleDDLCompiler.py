from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
class OracleDDLCompiler(compiler.DDLCompiler):

    def define_constraint_cascades(self, constraint):
        text = ''
        if constraint.ondelete is not None:
            text += ' ON DELETE %s' % constraint.ondelete
        if constraint.onupdate is not None:
            util.warn("Oracle does not contain native UPDATE CASCADE functionality - onupdates will not be rendered for foreign keys.  Consider using deferrable=True, initially='deferred' or triggers.")
        return text

    def visit_drop_table_comment(self, drop, **kw):
        return "COMMENT ON TABLE %s IS ''" % self.preparer.format_table(drop.element)

    def visit_create_index(self, create, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        if index.dialect_options['oracle']['bitmap']:
            text += 'BITMAP '
        text += 'INDEX %s ON %s (%s)' % (self._prepared_index_name(index, include_schema=True), preparer.format_table(index.table, use_schema=True), ', '.join((self.sql_compiler.process(expr, include_table=False, literal_binds=True) for expr in index.expressions)))
        if index.dialect_options['oracle']['compress'] is not False:
            if index.dialect_options['oracle']['compress'] is True:
                text += ' COMPRESS'
            else:
                text += ' COMPRESS %d' % index.dialect_options['oracle']['compress']
        return text

    def post_create_table(self, table):
        table_opts = []
        opts = table.dialect_options['oracle']
        if opts['on_commit']:
            on_commit_options = opts['on_commit'].replace('_', ' ').upper()
            table_opts.append('\n ON COMMIT %s' % on_commit_options)
        if opts['compress']:
            if opts['compress'] is True:
                table_opts.append('\n COMPRESS')
            else:
                table_opts.append('\n COMPRESS FOR %s' % opts['compress'])
        return ''.join(table_opts)

    def get_identity_options(self, identity_options):
        text = super().get_identity_options(identity_options)
        text = text.replace('NO MINVALUE', 'NOMINVALUE')
        text = text.replace('NO MAXVALUE', 'NOMAXVALUE')
        text = text.replace('NO CYCLE', 'NOCYCLE')
        if identity_options.order is not None:
            text += ' ORDER' if identity_options.order else ' NOORDER'
        return text.strip()

    def visit_computed_column(self, generated, **kw):
        text = 'GENERATED ALWAYS AS (%s)' % self.sql_compiler.process(generated.sqltext, include_table=False, literal_binds=True)
        if generated.persisted is True:
            raise exc.CompileError("Oracle computed columns do not support 'stored' persistence; set the 'persisted' flag to None or False for Oracle support.")
        elif generated.persisted is False:
            text += ' VIRTUAL'
        return text

    def visit_identity_column(self, identity, **kw):
        if identity.always is None:
            kind = ''
        else:
            kind = 'ALWAYS' if identity.always else 'BY DEFAULT'
        text = 'GENERATED %s' % kind
        if identity.on_null:
            text += ' ON NULL'
        text += ' AS IDENTITY'
        options = self.get_identity_options(identity)
        if options:
            text += ' (%s)' % options
        return text