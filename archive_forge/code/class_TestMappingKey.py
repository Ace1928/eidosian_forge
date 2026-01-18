import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestMappingKey:

    def test_simple_mapping_key(self):
        inp = '        {a: 1, b: 2}: hello world\n        '
        round_trip(inp, preserve_quotes=True, dump_data=False)

    def test_set_simple_mapping_key(self):
        from srsly.ruamel_yaml.comments import CommentedKeyMap
        d = {CommentedKeyMap([('a', 1), ('b', 2)]): 'hello world'}
        exp = dedent('        {a: 1, b: 2}: hello world\n        ')
        assert round_trip_dump(d) == exp

    def test_change_key_simple_mapping_key(self):
        from srsly.ruamel_yaml.comments import CommentedKeyMap
        inp = '        {a: 1, b: 2}: hello world\n        '
        d = round_trip_load(inp, preserve_quotes=True)
        d[CommentedKeyMap([('b', 1), ('a', 2)])] = d.pop(CommentedKeyMap([('a', 1), ('b', 2)]))
        exp = dedent('        {b: 1, a: 2}: hello world\n        ')
        assert round_trip_dump(d) == exp

    def test_change_value_simple_mapping_key(self):
        from srsly.ruamel_yaml.comments import CommentedKeyMap
        inp = '        {a: 1, b: 2}: hello world\n        '
        d = round_trip_load(inp, preserve_quotes=True)
        d = {CommentedKeyMap([('a', 1), ('b', 2)]): 'goodbye'}
        exp = dedent('        {a: 1, b: 2}: goodbye\n        ')
        assert round_trip_dump(d) == exp