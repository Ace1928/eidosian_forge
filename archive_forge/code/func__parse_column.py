import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_column(self, line, state):
    """Extract column details.

        Falls back to a 'minimal support' variant if full parse fails.

        :param line: Any column-bearing line from SHOW CREATE TABLE
        """
    spec = None
    m = self._re_column.match(line)
    if m:
        spec = m.groupdict()
        spec['full'] = True
    else:
        m = self._re_column_loose.match(line)
        if m:
            spec = m.groupdict()
            spec['full'] = False
    if not spec:
        util.warn('Unknown column definition %r' % line)
        return
    if not spec['full']:
        util.warn('Incomplete reflection of column definition %r' % line)
    name, type_, args = (spec['name'], spec['coltype'], spec['arg'])
    try:
        col_type = self.dialect.ischema_names[type_]
    except KeyError:
        util.warn("Did not recognize type '%s' of column '%s'" % (type_, name))
        col_type = sqltypes.NullType
    if args is None or args == '':
        type_args = []
    elif args[0] == "'" and args[-1] == "'":
        type_args = self._re_csv_str.findall(args)
    else:
        type_args = [int(v) for v in self._re_csv_int.findall(args)]
    type_kw = {}
    if issubclass(col_type, (DATETIME, TIME, TIMESTAMP)):
        if type_args:
            type_kw['fsp'] = type_args.pop(0)
    for kw in ('unsigned', 'zerofill'):
        if spec.get(kw, False):
            type_kw[kw] = True
    for kw in ('charset', 'collate'):
        if spec.get(kw, False):
            type_kw[kw] = spec[kw]
    if issubclass(col_type, (ENUM, SET)):
        type_args = _strip_values(type_args)
        if issubclass(col_type, SET) and '' in type_args:
            type_kw['retrieve_as_bitwise'] = True
    type_instance = col_type(*type_args, **type_kw)
    col_kw = {}
    col_kw['nullable'] = True
    if spec.get('notnull', False) == 'NOT NULL':
        col_kw['nullable'] = False
    if spec.get('notnull_generated', False) == 'NOT NULL':
        col_kw['nullable'] = False
    if spec.get('autoincr', False):
        col_kw['autoincrement'] = True
    elif issubclass(col_type, sqltypes.Integer):
        col_kw['autoincrement'] = False
    default = spec.get('default', None)
    if default == 'NULL':
        default = None
    comment = spec.get('comment', None)
    if comment is not None:
        comment = cleanup_text(comment)
    sqltext = spec.get('generated')
    if sqltext is not None:
        computed = dict(sqltext=sqltext)
        persisted = spec.get('persistence')
        if persisted is not None:
            computed['persisted'] = persisted == 'STORED'
        col_kw['computed'] = computed
    col_d = dict(name=name, type=type_instance, default=default, comment=comment)
    col_d.update(col_kw)
    state.columns.append(col_d)