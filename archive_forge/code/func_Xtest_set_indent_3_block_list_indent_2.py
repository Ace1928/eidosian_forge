from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def Xtest_set_indent_3_block_list_indent_2(self):
    inp = '\n        a:\n          -\n           b: c\n          -\n           1\n          -\n           d:\n             -\n              2\n        '
    round_trip(inp, indent=3, block_seq_indent=2)