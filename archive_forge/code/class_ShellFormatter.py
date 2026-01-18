from . import base
from cliff import columns
import argparse
class ShellFormatter(base.SingleFormatter):

    def add_argument_group(self, parser):
        group = parser.add_argument_group(title='shell formatter', description='a format a UNIX shell can parse (variable="value")')
        group.add_argument('--variable', action='append', default=[], dest='variables', metavar='VARIABLE', help=argparse.SUPPRESS)
        group.add_argument('--prefix', action='store', default='', dest='prefix', help='add a prefix to all variable names')

    def emit_one(self, column_names, data, stdout, parsed_args):
        variable_names = [c.lower().replace(' ', '_') for c in column_names]
        desired_columns = parsed_args.variables
        for name, value in zip(variable_names, data):
            if name in desired_columns or not desired_columns:
                value = str(value.machine_readable()) if isinstance(value, columns.FormattableColumn) else value
                if isinstance(value, str):
                    value = value.replace('"', '\\"')
                if isinstance(name, str):
                    name = name.replace(':', '_')
                    name = name.replace('-', '_')
                stdout.write('%s%s="%s"\n' % (parsed_args.prefix, name, value))
        return