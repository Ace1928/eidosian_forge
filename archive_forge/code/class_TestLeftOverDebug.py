import sys
import pytest  # NOQA
from .roundtrip import round_trip_load, round_trip_dump, dedent
class TestLeftOverDebug:

    def test_00(self, capsys):
        s = dedent('\n        a: 1\n        b: []\n        c: [a, 1]\n        d: {f: 3.14, g: 42}\n        ')
        d = round_trip_load(s)
        round_trip_dump(d, sys.stdout)
        out, err = capsys.readouterr()
        assert out == s

    def test_01(self, capsys):
        s = dedent('\n        - 1\n        - []\n        - [a, 1]\n        - {f: 3.14, g: 42}\n        - - 123\n        ')
        d = round_trip_load(s)
        round_trip_dump(d, sys.stdout)
        out, err = capsys.readouterr()
        assert out == s