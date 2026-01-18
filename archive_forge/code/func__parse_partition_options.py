import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_partition_options(self, line, state):
    options = {}
    new_line = line[:]
    while new_line.startswith('(') or new_line.startswith(' '):
        new_line = new_line[1:]
    for regex, cleanup in self._pr_options:
        m = regex.search(new_line)
        if not m or 'PARTITION' not in regex.pattern:
            continue
        directive = m.group('directive')
        directive = directive.lower()
        is_subpartition = directive == 'subpartition'
        if directive == 'partition' or is_subpartition:
            new_line = new_line.replace(') */', '')
            new_line = new_line.replace(',', '')
            if is_subpartition and new_line.endswith(')'):
                new_line = new_line[:-1]
            if self.dialect.name == 'mariadb' and new_line.endswith(')'):
                if 'MAXVALUE' in new_line or 'MINVALUE' in new_line or 'ENGINE' in new_line:
                    new_line = new_line[:-1]
            defs = '%s_%s_definitions' % (self.dialect.name, directive)
            options[defs] = new_line
        else:
            directive = directive.replace(' ', '_')
            value = m.group('val')
            if cleanup:
                value = cleanup(value)
            options[directive] = value
        break
    for opt, val in options.items():
        part_def = '%s_partition_definitions' % self.dialect.name
        subpart_def = '%s_subpartition_definitions' % self.dialect.name
        if opt == part_def or opt == subpart_def:
            if opt not in state.table_options:
                state.table_options[opt] = val
            else:
                state.table_options[opt] = '%s, %s' % (state.table_options[opt], val)
        else:
            state.table_options['%s_%s' % (self.dialect.name, opt)] = val