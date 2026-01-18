from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
class IdentifierPreparer:
    """Handle quoting and case-folding of identifiers based on options."""
    reserved_words = RESERVED_WORDS
    legal_characters = LEGAL_CHARACTERS
    illegal_initial_characters = ILLEGAL_INITIAL_CHARACTERS
    initial_quote: str
    final_quote: str
    _strings: MutableMapping[str, str]
    schema_for_object: _SchemaForObjectCallable = operator.attrgetter('schema')
    'Return the .schema attribute for an object.\n\n    For the default IdentifierPreparer, the schema for an object is always\n    the value of the ".schema" attribute.   if the preparer is replaced\n    with one that has a non-empty schema_translate_map, the value of the\n    ".schema" attribute is rendered a symbol that will be converted to a\n    real schema name from the mapping post-compile.\n\n    '
    _includes_none_schema_translate: bool = False

    def __init__(self, dialect, initial_quote='"', final_quote=None, escape_quote='"', quote_case_sensitive_collations=True, omit_schema=False):
        """Construct a new ``IdentifierPreparer`` object.

        initial_quote
          Character that begins a delimited identifier.

        final_quote
          Character that ends a delimited identifier. Defaults to
          `initial_quote`.

        omit_schema
          Prevent prepending schema name. Useful for databases that do
          not support schemae.
        """
        self.dialect = dialect
        self.initial_quote = initial_quote
        self.final_quote = final_quote or self.initial_quote
        self.escape_quote = escape_quote
        self.escape_to_quote = self.escape_quote * 2
        self.omit_schema = omit_schema
        self.quote_case_sensitive_collations = quote_case_sensitive_collations
        self._strings = {}
        self._double_percents = self.dialect.paramstyle in ('format', 'pyformat')

    def _with_schema_translate(self, schema_translate_map):
        prep = self.__class__.__new__(self.__class__)
        prep.__dict__.update(self.__dict__)
        includes_none = None in schema_translate_map

        def symbol_getter(obj):
            name = obj.schema
            if obj._use_schema_map and (name is not None or includes_none):
                if name is not None and ('[' in name or ']' in name):
                    raise exc.CompileError("Square bracket characters ([]) not supported in schema translate name '%s'" % name)
                return quoted_name('__[SCHEMA_%s]' % (name or '_none'), quote=False)
            else:
                return obj.schema
        prep.schema_for_object = symbol_getter
        prep._includes_none_schema_translate = includes_none
        return prep

    def _render_schema_translates(self, statement, schema_translate_map):
        d = schema_translate_map
        if None in d:
            if not self._includes_none_schema_translate:
                raise exc.InvalidRequestError('schema translate map which previously did not have `None` present as a key now has `None` present; compiled statement may lack adequate placeholders.  Please use consistent keys in successive schema_translate_map dictionaries.')
            d['_none'] = d[None]

        def replace(m):
            name = m.group(2)
            if name in d:
                effective_schema = d[name]
            else:
                if name in (None, '_none'):
                    raise exc.InvalidRequestError("schema translate map which previously had `None` present as a key now no longer has it present; don't know how to apply schema for compiled statement. Please use consistent keys in successive schema_translate_map dictionaries.")
                effective_schema = name
            if not effective_schema:
                effective_schema = self.dialect.default_schema_name
                if not effective_schema:
                    raise exc.CompileError("Dialect has no default schema name; can't use None as dynamic schema target.")
            return self.quote_schema(effective_schema)
        return re.sub('(__\\[SCHEMA_([^\\]]+)\\])', replace, statement)

    def _escape_identifier(self, value: str) -> str:
        """Escape an identifier.

        Subclasses should override this to provide database-dependent
        escaping behavior.
        """
        value = value.replace(self.escape_quote, self.escape_to_quote)
        if self._double_percents:
            value = value.replace('%', '%%')
        return value

    def _unescape_identifier(self, value: str) -> str:
        """Canonicalize an escaped identifier.

        Subclasses should override this to provide database-dependent
        unescaping behavior that reverses _escape_identifier.
        """
        return value.replace(self.escape_to_quote, self.escape_quote)

    def validate_sql_phrase(self, element, reg):
        """keyword sequence filter.

        a filter for elements that are intended to represent keyword sequences,
        such as "INITIALLY", "INITIALLY DEFERRED", etc.   no special characters
        should be present.

        .. versionadded:: 1.3

        """
        if element is not None and (not reg.match(element)):
            raise exc.CompileError('Unexpected SQL phrase: %r (matching against %r)' % (element, reg.pattern))
        return element

    def quote_identifier(self, value: str) -> str:
        """Quote an identifier.

        Subclasses should override this to provide database-dependent
        quoting behavior.
        """
        return self.initial_quote + self._escape_identifier(value) + self.final_quote

    def _requires_quotes(self, value: str) -> bool:
        """Return True if the given identifier requires quoting."""
        lc_value = value.lower()
        return lc_value in self.reserved_words or value[0] in self.illegal_initial_characters or (not self.legal_characters.match(str(value))) or (lc_value != value)

    def _requires_quotes_illegal_chars(self, value):
        """Return True if the given identifier requires quoting, but
        not taking case convention into account."""
        return not self.legal_characters.match(str(value))

    def quote_schema(self, schema: str, force: Any=None) -> str:
        """Conditionally quote a schema name.


        The name is quoted if it is a reserved word, contains quote-necessary
        characters, or is an instance of :class:`.quoted_name` which includes
        ``quote`` set to ``True``.

        Subclasses can override this to provide database-dependent
        quoting behavior for schema names.

        :param schema: string schema name
        :param force: unused

            .. deprecated:: 0.9

                The :paramref:`.IdentifierPreparer.quote_schema.force`
                parameter is deprecated and will be removed in a future
                release.  This flag has no effect on the behavior of the
                :meth:`.IdentifierPreparer.quote` method; please refer to
                :class:`.quoted_name`.

        """
        if force is not None:
            util.warn_deprecated('The IdentifierPreparer.quote_schema.force parameter is deprecated and will be removed in a future release.  This flag has no effect on the behavior of the IdentifierPreparer.quote method; please refer to quoted_name().', version='0.9')
        return self.quote(schema)

    def quote(self, ident: str, force: Any=None) -> str:
        """Conditionally quote an identifier.

        The identifier is quoted if it is a reserved word, contains
        quote-necessary characters, or is an instance of
        :class:`.quoted_name` which includes ``quote`` set to ``True``.

        Subclasses can override this to provide database-dependent
        quoting behavior for identifier names.

        :param ident: string identifier
        :param force: unused

            .. deprecated:: 0.9

                The :paramref:`.IdentifierPreparer.quote.force`
                parameter is deprecated and will be removed in a future
                release.  This flag has no effect on the behavior of the
                :meth:`.IdentifierPreparer.quote` method; please refer to
                :class:`.quoted_name`.

        """
        if force is not None:
            util.warn_deprecated('The IdentifierPreparer.quote.force parameter is deprecated and will be removed in a future release.  This flag has no effect on the behavior of the IdentifierPreparer.quote method; please refer to quoted_name().', version='0.9')
        force = getattr(ident, 'quote', None)
        if force is None:
            if ident in self._strings:
                return self._strings[ident]
            else:
                if self._requires_quotes(ident):
                    self._strings[ident] = self.quote_identifier(ident)
                else:
                    self._strings[ident] = ident
                return self._strings[ident]
        elif force:
            return self.quote_identifier(ident)
        else:
            return ident

    def format_collation(self, collation_name):
        if self.quote_case_sensitive_collations:
            return self.quote(collation_name)
        else:
            return collation_name

    def format_sequence(self, sequence, use_schema=True):
        name = self.quote(sequence.name)
        effective_schema = self.schema_for_object(sequence)
        if not self.omit_schema and use_schema and (effective_schema is not None):
            name = self.quote_schema(effective_schema) + '.' + name
        return name

    def format_label(self, label: Label[Any], name: Optional[str]=None) -> str:
        return self.quote(name or label.name)

    def format_alias(self, alias: Optional[AliasedReturnsRows], name: Optional[str]=None) -> str:
        if name is None:
            assert alias is not None
            return self.quote(alias.name)
        else:
            return self.quote(name)

    def format_savepoint(self, savepoint, name=None):
        ident = name or savepoint.ident
        if self._requires_quotes(ident):
            ident = self.quote_identifier(ident)
        return ident

    @util.preload_module('sqlalchemy.sql.naming')
    def format_constraint(self, constraint, _alembic_quote=True):
        naming = util.preloaded.sql_naming
        if constraint.name is _NONE_NAME:
            name = naming._constraint_name_for_table(constraint, constraint.table)
            if name is None:
                return None
        else:
            name = constraint.name
        if constraint.__visit_name__ == 'index':
            return self.truncate_and_render_index_name(name, _alembic_quote=_alembic_quote)
        else:
            return self.truncate_and_render_constraint_name(name, _alembic_quote=_alembic_quote)

    def truncate_and_render_index_name(self, name, _alembic_quote=True):
        max_ = self.dialect.max_index_name_length or self.dialect.max_identifier_length
        return self._truncate_and_render_maxlen_name(name, max_, _alembic_quote)

    def truncate_and_render_constraint_name(self, name, _alembic_quote=True):
        max_ = self.dialect.max_constraint_name_length or self.dialect.max_identifier_length
        return self._truncate_and_render_maxlen_name(name, max_, _alembic_quote)

    def _truncate_and_render_maxlen_name(self, name, max_, _alembic_quote):
        if isinstance(name, elements._truncated_label):
            if len(name) > max_:
                name = name[0:max_ - 8] + '_' + util.md5_hex(name)[-4:]
        else:
            self.dialect.validate_identifier(name)
        if not _alembic_quote:
            return name
        else:
            return self.quote(name)

    def format_index(self, index):
        return self.format_constraint(index)

    def format_table(self, table, use_schema=True, name=None):
        """Prepare a quoted table and schema name."""
        if name is None:
            name = table.name
        result = self.quote(name)
        effective_schema = self.schema_for_object(table)
        if not self.omit_schema and use_schema and effective_schema:
            result = self.quote_schema(effective_schema) + '.' + result
        return result

    def format_schema(self, name):
        """Prepare a quoted schema name."""
        return self.quote(name)

    def format_label_name(self, name, anon_map=None):
        """Prepare a quoted column name."""
        if anon_map is not None and isinstance(name, elements._truncated_label):
            name = name.apply_map(anon_map)
        return self.quote(name)

    def format_column(self, column, use_table=False, name=None, table_name=None, use_schema=False, anon_map=None):
        """Prepare a quoted column name."""
        if name is None:
            name = column.name
        if anon_map is not None and isinstance(name, elements._truncated_label):
            name = name.apply_map(anon_map)
        if not getattr(column, 'is_literal', False):
            if use_table:
                return self.format_table(column.table, use_schema=use_schema, name=table_name) + '.' + self.quote(name)
            else:
                return self.quote(name)
        elif use_table:
            return self.format_table(column.table, use_schema=use_schema, name=table_name) + '.' + name
        else:
            return name

    def format_table_seq(self, table, use_schema=True):
        """Format table name and schema as a tuple."""
        effective_schema = self.schema_for_object(table)
        if not self.omit_schema and use_schema and effective_schema:
            return (self.quote_schema(effective_schema), self.format_table(table, use_schema=False))
        else:
            return (self.format_table(table, use_schema=False),)

    @util.memoized_property
    def _r_identifiers(self):
        initial, final, escaped_final = (re.escape(s) for s in (self.initial_quote, self.final_quote, self._escape_identifier(self.final_quote)))
        r = re.compile('(?:(?:%(initial)s((?:%(escaped)s|[^%(final)s])+)%(final)s|([^\\.]+))(?=\\.|$))+' % {'initial': initial, 'final': final, 'escaped': escaped_final})
        return r

    def unformat_identifiers(self, identifiers):
        """Unpack 'schema.table.column'-like strings into components."""
        r = self._r_identifiers
        return [self._unescape_identifier(i) for i in [a or b for a, b in r.findall(identifiers)]]