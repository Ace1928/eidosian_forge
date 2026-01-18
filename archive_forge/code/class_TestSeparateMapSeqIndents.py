from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
class TestSeparateMapSeqIndents:

    def test_00(self):
        yaml = YAML()
        yaml.indent = 6
        yaml.block_seq_indent = 3
        inp = '\n        a:\n           -  1\n           -  [1, 2]\n        '
        yaml.round_trip(inp)

    def test_01(self):
        yaml = YAML()
        yaml.indent(sequence=6)
        yaml.indent(offset=3)
        inp = '\n        a:\n           -  1\n           -  {b: 3}\n        '
        yaml.round_trip(inp)

    def test_02(self):
        yaml = YAML()
        yaml.indent(mapping=5, sequence=6, offset=3)
        inp = '\n        a:\n             b:\n                -  1\n                -  [1, 2]\n        '
        yaml.round_trip(inp)

    def test_03(self):
        inp = '\n        a:\n            b:\n                c:\n                -   1\n                -   [1, 2]\n        '
        round_trip(inp, indent=4)

    def test_04(self):
        yaml = YAML()
        yaml.indent(mapping=5, sequence=6)
        inp = '\n        a:\n             b:\n             -     1\n             -     [1, 2]\n             -     {d: 3.14}\n        '
        yaml.round_trip(inp)

    def test_issue_51(self):
        yaml = YAML()
        yaml.indent(sequence=4, offset=2)
        yaml.preserve_quotes = True
        yaml.round_trip("\n        role::startup::author::rsyslog_inputs:\n          imfile:\n            - ruleset: 'AEM-slinglog'\n              File: '/opt/aem/author/crx-quickstart/logs/error.log'\n              startmsg.regex: '^[-+T.:[:digit:]]*'\n              tag: 'error'\n            - ruleset: 'AEM-slinglog'\n              File: '/opt/aem/author/crx-quickstart/logs/stdout.log'\n              startmsg.regex: '^[-+T.:[:digit:]]*'\n              tag: 'stdout'\n        ")