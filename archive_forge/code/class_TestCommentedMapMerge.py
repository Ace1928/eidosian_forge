import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestCommentedMapMerge:

    def test_in_operator(self):
        data = round_trip_load('\n        x: &base\n          a: 1\n          b: 2\n          c: 3\n        y:\n          <<: *base\n          k: 4\n          l: 5\n        ')
        assert data['x']['a'] == 1
        assert 'a' in data['x']
        assert data['y']['a'] == 1
        assert 'a' in data['y']

    def test_issue_60(self):
        data = round_trip_load('\n        x: &base\n          a: 1\n        y:\n          <<: *base\n        ')
        assert data['x']['a'] == 1
        assert data['y']['a'] == 1
        if sys.version_info >= (3, 12):
            assert str(data['y']) == "ordereddict({'a': 1})"
        else:
            assert str(data['y']) == "ordereddict([('a', 1)])"

    def test_issue_60_1(self):
        data = round_trip_load('\n        x: &base\n          a: 1\n        y:\n          <<: *base\n          b: 2\n        ')
        assert data['x']['a'] == 1
        assert data['y']['a'] == 1
        if sys.version_info >= (3, 12):
            assert str(data['y']) == "ordereddict({'b': 2, 'a': 1})"
        else:
            assert str(data['y']) == "ordereddict([('b', 2), ('a', 1)])"