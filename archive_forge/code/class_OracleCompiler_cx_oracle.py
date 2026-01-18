from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
class OracleCompiler_cx_oracle(OracleCompiler):
    _oracle_cx_sql_compiler = True
    _oracle_returning = False
    bindname_escape_characters = util.immutabledict({'%': 'P', '(': 'A', ')': 'Z', ':': 'C', '.': 'C', '[': 'C', ']': 'C', ' ': 'C', '\\': 'C', '/': 'C', '?': 'C'})

    def bindparam_string(self, name, **kw):
        quote = getattr(name, 'quote', None)
        if quote is True or (quote is not False and self.preparer._bindparam_requires_quotes(name) and (not kw.get('post_compile', False))):
            quoted_name = '"%s"' % name
            kw['escaped_from'] = name
            name = quoted_name
            return OracleCompiler.bindparam_string(self, name, **kw)
        escaped_from = kw.get('escaped_from', None)
        if not escaped_from:
            if self._bind_translate_re.search(name):
                new_name = self._bind_translate_re.sub(lambda m: self._bind_translate_chars[m.group(0)], name)
                if new_name[0].isdigit() or new_name[0] == '_':
                    new_name = 'D' + new_name
                kw['escaped_from'] = name
                name = new_name
            elif name[0].isdigit() or name[0] == '_':
                new_name = 'D' + name
                kw['escaped_from'] = name
                name = new_name
        return OracleCompiler.bindparam_string(self, name, **kw)