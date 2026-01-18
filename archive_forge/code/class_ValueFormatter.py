from . import base
from cliff import columns
class ValueFormatter(base.ListFormatter, base.SingleFormatter):

    def add_argument_group(self, parser):
        pass

    def emit_list(self, column_names, data, stdout, parsed_args):
        for row in data:
            stdout.write(' '.join((str(c.machine_readable() if isinstance(c, columns.FormattableColumn) else c) for c in row)) + '\n')
        return

    def emit_one(self, column_names, data, stdout, parsed_args):
        for value in data:
            stdout.write('%s\n' % str(value.machine_readable() if isinstance(value, columns.FormattableColumn) else value))
        return