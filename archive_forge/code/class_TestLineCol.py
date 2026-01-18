import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestLineCol:

    def test_item_00(self):
        data = load('\n            - a\n            - e\n            - [b, d]\n            - c\n            ')
        assert data[2].lc.line == 2
        assert data[2].lc.col == 2

    def test_item_01(self):
        data = load('\n            - a\n            - e\n            - {x: 3}\n            - c\n            ')
        assert data[2].lc.line == 2
        assert data[2].lc.col == 2

    def test_item_02(self):
        data = load('\n            - a\n            - e\n            - !!set {x, y}\n            - c\n            ')
        assert data[2].lc.line == 2
        assert data[2].lc.col == 2

    def test_item_03(self):
        data = load('\n            - a\n            - e\n            - !!omap\n              - x: 1\n              - y: 3\n            - c\n            ')
        assert data[2].lc.line == 2
        assert data[2].lc.col == 2

    def test_item_04(self):
        data = load('\n         # testing line and column based on SO\n         # http://stackoverflow.com/questions/13319067/\n         - key1: item 1\n           key2: item 2\n         - key3: another item 1\n           key4: another item 2\n            ')
        assert data[0].lc.line == 2
        assert data[0].lc.col == 2
        assert data[1].lc.line == 4
        assert data[1].lc.col == 2

    def test_pos_mapping(self):
        data = load('\n        a: 1\n        b: 2\n        c: 3\n        # comment\n        klm: 42\n        d: 4\n        ')
        assert data.lc.key('klm') == (4, 0)
        assert data.lc.value('klm') == (4, 5)

    def test_pos_sequence(self):
        data = load('\n        - a\n        - b\n        - c\n        # next one!\n        - klm\n        - d\n        ')
        assert data.lc.item(3) == (4, 2)