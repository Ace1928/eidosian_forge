import collections
import hashlib
from cliff.formatters import base
class ResourceDotFormatter(base.ListFormatter):

    def add_argument_group(self, parser):
        pass

    def emit_list(self, column_names, data, stdout, parsed_args):
        writer = ResourceDotWriter(data, stdout)
        writer.write()