from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
class TestNoIndent:

    def test_root_literal_scalar_indent_example_9_5(self):
        yaml = YAML()
        s = '%!PS-Adobe-2.0'
        inp = '\n        --- |\n          {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_no_indent(self):
        yaml = YAML()
        s = 'testing123'
        inp = '\n        --- |\n        {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_no_indent_1_1(self):
        yaml = YAML()
        s = 'testing123'
        inp = '\n        %YAML 1.1\n        --- |\n        {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_no_indent_1_1_old_style(self):
        from textwrap import dedent
        from srsly.ruamel_yaml import safe_load
        s = 'testing123'
        inp = '\n        %YAML 1.1\n        --- |\n          {}\n        '
        d = safe_load(dedent(inp.format(s)))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_no_indent_1_1_no_raise(self):
        yaml = YAML()
        yaml.root_level_block_style_scalar_no_indent_error_1_1 = True
        s = 'testing123'
        if True:
            inp = '\n            %YAML 1.1\n            --- |\n            {}\n            '
            yaml.load(inp.format(s))

    def test_root_literal_scalar_indent_offset_one(self):
        yaml = YAML()
        s = 'testing123'
        inp = '\n        --- |1\n         {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_indent_offset_four(self):
        yaml = YAML()
        s = 'testing123'
        inp = '\n        --- |4\n            {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_scalar_indent_offset_two_leading_space(self):
        yaml = YAML()
        s = ' testing123'
        inp = '\n        --- |4\n            {s}\n            {s}\n        '
        d = yaml.load(inp.format(s=s))
        print(d)
        assert d == (s + '\n') * 2

    def test_root_literal_scalar_no_indent_special(self):
        yaml = YAML()
        s = '%!PS-Adobe-2.0'
        inp = '\n        --- |\n        {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_folding_scalar_indent(self):
        yaml = YAML()
        s = '%!PS-Adobe-2.0'
        inp = '\n        --- >\n          {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_folding_scalar_no_indent(self):
        yaml = YAML()
        s = 'testing123'
        inp = '\n        --- >\n        {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_folding_scalar_no_indent_special(self):
        yaml = YAML()
        s = '%!PS-Adobe-2.0'
        inp = '\n        --- >\n        {}\n        '
        d = yaml.load(inp.format(s))
        print(d)
        assert d == s + '\n'

    def test_root_literal_multi_doc(self):
        yaml = YAML(typ='safe', pure=True)
        s1 = 'abc'
        s2 = 'klm'
        inp = '\n        --- |-\n        {}\n        --- |\n        {}\n        '
        for idx, d1 in enumerate(yaml.load_all(inp.format(s1, s2))):
            print('d1:', d1)
            assert ['abc', 'klm\n'][idx] == d1

    def test_root_literal_doc_indent_directives_end(self):
        yaml = YAML()
        yaml.explicit_start = True
        inp = '\n        --- |-\n          %YAML 1.3\n          ---\n          this: is a test\n        '
        yaml.round_trip(inp)

    def test_root_literal_doc_indent_document_end(self):
        yaml = YAML()
        yaml.explicit_start = True
        inp = '\n        --- |-\n          some more\n          ...\n          text\n        '
        yaml.round_trip(inp)

    def test_root_literal_doc_indent_marker(self):
        yaml = YAML()
        yaml.explicit_start = True
        inp = '\n        --- |2\n           some more\n          text\n        '
        d = yaml.load(inp)
        print(type(d), repr(d))
        yaml.round_trip(inp)

    def test_nested_literal_doc_indent_marker(self):
        yaml = YAML()
        yaml.explicit_start = True
        inp = '\n        ---\n        a: |2\n           some more\n          text\n        '
        d = yaml.load(inp)
        print(type(d), repr(d))
        yaml.round_trip(inp)