import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestEmptyLines:

    def test_issue_46(self):
        yaml_str = dedent("        ---\n        # Please add key/value pairs in alphabetical order\n\n        aws_s3_bucket: 'mys3bucket'\n\n        jenkins_ad_credentials:\n          bind_name: 'CN=svc-AAA-BBB-T,OU=Example,DC=COM,DC=EXAMPLE,DC=Local'\n          bind_pass: 'xxxxyyyy{'\n        ")
        d = round_trip_load(yaml_str, preserve_quotes=True)
        y = round_trip_dump(d, explicit_start=True)
        assert yaml_str == y

    def test_multispace_map(self):
        round_trip('\n        a: 1x\n\n        b: 2x\n\n\n        c: 3x\n\n\n\n        d: 4x\n\n        ')

    @pytest.mark.xfail(strict=True)
    def test_multispace_map_initial(self):
        round_trip('\n\n        a: 1x\n\n        b: 2x\n\n\n        c: 3x\n\n\n\n        d: 4x\n\n        ')

    def test_embedded_map(self):
        round_trip('\n        - a: 1y\n          b: 2y\n\n          c: 3y\n        ')

    def test_toplevel_seq(self):
        round_trip('        - 1\n\n        - 2\n\n        - 3\n        ')

    def test_embedded_seq(self):
        round_trip('\n        a:\n          b:\n          - 1\n\n          - 2\n\n\n          - 3\n        ')

    def test_line_with_only_spaces(self):
        yaml_str = "---\n\na: 'x'\n \nb: y\n"
        d = round_trip_load(yaml_str, preserve_quotes=True)
        y = round_trip_dump(d, explicit_start=True)
        stripped = ''
        for line in yaml_str.splitlines():
            stripped += line.rstrip() + '\n'
            print(line + '$')
        assert stripped == y

    def test_some_eol_spaces(self):
        yaml_str = '---  \n  \na: "x"  \n   \nb: y  \n'
        d = round_trip_load(yaml_str, preserve_quotes=True)
        y = round_trip_dump(d, explicit_start=True)
        stripped = ''
        for line in yaml_str.splitlines():
            stripped += line.rstrip() + '\n'
            print(line + '$')
        assert stripped == y

    def test_issue_54_not_ok(self):
        yaml_str = dedent('        toplevel:\n\n            # some comment\n            sublevel: 300\n        ')
        d = round_trip_load(yaml_str)
        print(d.ca)
        y = round_trip_dump(d, indent=4)
        print(y.replace('\n', '$\n'))
        assert yaml_str == y

    def test_issue_54_ok(self):
        yaml_str = dedent('        toplevel:\n            # some comment\n            sublevel: 300\n        ')
        d = round_trip_load(yaml_str)
        y = round_trip_dump(d, indent=4)
        assert yaml_str == y

    def test_issue_93(self):
        round_trip('        a:\n          b:\n          - c1: cat  # a1\n          # my comment on catfish\n          - c2: catfish  # a2\n        ')

    def test_issue_93_00(self):
        round_trip('        a:\n        - - c1: cat   # a1\n          # my comment on catfish\n          - c2: catfish  # a2\n        ')

    def test_issue_93_01(self):
        round_trip('        - - c1: cat   # a1\n          # my comment on catfish\n          - c2: catfish  # a2\n        ')

    def test_issue_93_02(self):
        round_trip('        - c1: cat\n        # my comment on catfish\n        - c2: catfish\n        ')

    def test_issue_96(self):
        round_trip('        a:\n          b:\n            c: c_val\n            d:\n\n          e:\n            g: g_val\n        ')