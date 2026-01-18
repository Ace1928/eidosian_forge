from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestWalkTree:

    def test_basic(self):
        from srsly.ruamel_yaml.comments import CommentedMap
        from srsly.ruamel_yaml.scalarstring import walk_tree
        data = CommentedMap()
        data[1] = 'a'
        data[2] = 'with\nnewline\n'
        walk_tree(data)
        exp = '        1: a\n        2: |\n          with\n          newline\n        '
        assert round_trip_dump(data) == dedent(exp)

    def test_map(self):
        from srsly.ruamel_yaml.compat import ordereddict
        from srsly.ruamel_yaml.comments import CommentedMap
        from srsly.ruamel_yaml.scalarstring import walk_tree, preserve_literal
        from srsly.ruamel_yaml.scalarstring import DoubleQuotedScalarString as dq
        from srsly.ruamel_yaml.scalarstring import SingleQuotedScalarString as sq
        data = CommentedMap()
        data[1] = 'a'
        data[2] = 'with\nnew : line\n'
        data[3] = '${abc}'
        data[4] = 'almost:mapping'
        m = ordereddict([('\n', preserve_literal), ('${', sq), (':', dq)])
        walk_tree(data, map=m)
        exp = '        1: a\n        2: |\n          with\n          new : line\n        3: \'${abc}\'\n        4: "almost:mapping"\n        '
        assert round_trip_dump(data) == dedent(exp)