from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import next, text_type
from petl.errors import FieldSelectionError
from petl.util.base import Table, asindices, rowgetter
class PushHeaderView(Table):

    def __init__(self, source, header, *args):
        self.source = source
        self.args = args
        if isinstance(header, (list, tuple)):
            self.header = header
        elif len(args) > 0:
            self.header = []
            self.header.append(header)
            self.header.extend(args)
        else:
            assert False, 'bad parameters'

    def __iter__(self):
        return iterpushheader(self.source, self.header)