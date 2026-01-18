from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestYAML:

    def test_backslash(self):
        round_trip('\n        handlers:\n          static_files: applications/\\1/static/\\2\n        ')

    def test_omap_out(self):
        from srsly.ruamel_yaml.compat import ordereddict
        import srsly.ruamel_yaml
        x = ordereddict([('a', 1), ('b', 2)])
        res = srsly.ruamel_yaml.dump(x, default_flow_style=False)
        assert res == dedent('\n        !!omap\n        - a: 1\n        - b: 2\n        ')

    def test_omap_roundtrip(self):
        round_trip('\n        !!omap\n        - a: 1\n        - b: 2\n        - c: 3\n        - d: 4\n        ')

    @pytest.mark.skipif(sys.version_info < (2, 7), reason='collections not available')
    def test_dump_collections_ordereddict(self):
        from collections import OrderedDict
        import srsly.ruamel_yaml
        x = OrderedDict([('a', 1), ('b', 2)])
        res = srsly.ruamel_yaml.dump(x, Dumper=srsly.ruamel_yaml.RoundTripDumper, default_flow_style=False)
        assert res == dedent('\n        !!omap\n        - a: 1\n        - b: 2\n        ')

    @pytest.mark.skipif(sys.version_info >= (3, 0) or platform.python_implementation() != 'CPython', reason='srsly.ruamel_yaml not available')
    def test_dump_ruamel_ordereddict(self):
        from srsly.ruamel_yaml.compat import ordereddict
        import srsly.ruamel_yaml
        x = ordereddict([('a', 1), ('b', 2)])
        res = srsly.ruamel_yaml.dump(x, Dumper=srsly.ruamel_yaml.RoundTripDumper, default_flow_style=False)
        assert res == dedent('\n        !!omap\n        - a: 1\n        - b: 2\n        ')

    def test_CommentedSet(self):
        from srsly.ruamel_yaml.constructor import CommentedSet
        s = CommentedSet(['a', 'b', 'c'])
        s.remove('b')
        s.add('d')
        assert s == CommentedSet(['a', 'c', 'd'])
        s.add('e')
        s.add('f')
        s.remove('e')
        assert s == CommentedSet(['a', 'c', 'd', 'f'])

    def test_set_out(self):
        import srsly.ruamel_yaml
        x = set(['a', 'b', 'c'])
        res = srsly.ruamel_yaml.dump(x, default_flow_style=False)
        assert res == dedent('\n        !!set\n        a: null\n        b: null\n        c: null\n        ')

    def test_set_compact(self):
        round_trip('\n        !!set\n        ? a\n        ? b\n        ? c\n        ')

    def test_blank_line_after_comment(self):
        round_trip('\n        # Comment with spaces after it.\n\n\n        a: 1\n        ')

    def test_blank_line_between_seq_items(self):
        round_trip('\n        # Seq with empty lines in between items.\n        b:\n        - bar\n\n\n        - baz\n        ')

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_blank_line_after_literal_chip(self):
        s = '\n        c:\n        - |\n          This item\n          has a blank line\n          following it.\n\n        - |\n          To visually separate it from this item.\n\n          This item contains a blank line.\n\n\n        '
        d = round_trip_load(dedent(s))
        print(d)
        round_trip(s)
        assert d['c'][0].split('it.')[1] == '\n'
        assert d['c'][1].split('line.')[1] == '\n'

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_blank_line_after_literal_keep(self):
        """ have to insert an eof marker in YAML to test this"""
        s = '\n        c:\n        - |+\n          This item\n          has a blank line\n          following it.\n\n        - |+\n          To visually separate it from this item.\n\n          This item contains a blank line.\n\n\n        ...\n        '
        d = round_trip_load(dedent(s))
        print(d)
        round_trip(s)
        assert d['c'][0].split('it.')[1] == '\n\n'
        assert d['c'][1].split('line.')[1] == '\n\n\n'

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_blank_line_after_literal_strip(self):
        s = '\n        c:\n        - |-\n          This item\n          has a blank line\n          following it.\n\n        - |-\n          To visually separate it from this item.\n\n          This item contains a blank line.\n\n\n        '
        d = round_trip_load(dedent(s))
        print(d)
        round_trip(s)
        assert d['c'][0].split('it.')[1] == ''
        assert d['c'][1].split('line.')[1] == ''

    def test_load_all_perserve_quotes(self):
        import srsly.ruamel_yaml
        s = dedent('        a: \'hello\'\n        ---\n        b: "goodbye"\n        ')
        data = []
        for x in srsly.ruamel_yaml.round_trip_load_all(s, preserve_quotes=True):
            data.append(x)
        out = srsly.ruamel_yaml.dump_all(data, Dumper=srsly.ruamel_yaml.RoundTripDumper)
        print(type(data[0]['a']), data[0]['a'])
        print(out)
        assert out == s