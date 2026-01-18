import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_constraints(self, line):
    """Parse a KEY or CONSTRAINT line.

        :param line: A line of SHOW CREATE TABLE output
        """
    m = self._re_key.match(line)
    if m:
        spec = m.groupdict()
        spec['columns'] = self._parse_keyexprs(spec['columns'])
        if spec['version_sql']:
            m2 = self._re_key_version_sql.match(spec['version_sql'])
            if m2 and m2.groupdict()['parser']:
                spec['parser'] = m2.groupdict()['parser']
        if spec['parser']:
            spec['parser'] = self.preparer.unformat_identifiers(spec['parser'])[0]
        return ('key', spec)
    m = self._re_fk_constraint.match(line)
    if m:
        spec = m.groupdict()
        spec['table'] = self.preparer.unformat_identifiers(spec['table'])
        spec['local'] = [c[0] for c in self._parse_keyexprs(spec['local'])]
        spec['foreign'] = [c[0] for c in self._parse_keyexprs(spec['foreign'])]
        return ('fk_constraint', spec)
    m = self._re_ck_constraint.match(line)
    if m:
        spec = m.groupdict()
        return ('ck_constraint', spec)
    m = self._re_partition.match(line)
    if m:
        return ('partition', line)
    return (None, line)