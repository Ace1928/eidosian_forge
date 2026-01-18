import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
class TestMergeKeysValues:
    yaml_str = dedent('    - &mx\n      a: x1\n      b: x2\n      c: x3\n    - &my\n      a: y1\n      b: y2  # masked by the one in &mx\n      d: y4\n    -\n      a: 1\n      <<: [*mx, *my]\n      m: 6\n    ')

    def test_merge_for(self):
        from srsly.ruamel_yaml import safe_load
        d = safe_load(self.yaml_str)
        data = round_trip_load(self.yaml_str)
        count = 0
        for x in data[2]:
            count += 1
            print(count, x)
        assert count == len(d[2])

    def test_merge_keys(self):
        from srsly.ruamel_yaml import safe_load
        d = safe_load(self.yaml_str)
        data = round_trip_load(self.yaml_str)
        count = 0
        for x in data[2].keys():
            count += 1
            print(count, x)
        assert count == len(d[2])

    def test_merge_values(self):
        from srsly.ruamel_yaml import safe_load
        d = safe_load(self.yaml_str)
        data = round_trip_load(self.yaml_str)
        count = 0
        for x in data[2].values():
            count += 1
            print(count, x)
        assert count == len(d[2])

    def test_merge_items(self):
        from srsly.ruamel_yaml import safe_load
        d = safe_load(self.yaml_str)
        data = round_trip_load(self.yaml_str)
        count = 0
        for x in data[2].items():
            count += 1
            print(count, x)
        assert count == len(d[2])

    def test_len_items_delete(self):
        from srsly.ruamel_yaml import safe_load
        from srsly.ruamel_yaml.compat import PY3
        d = safe_load(self.yaml_str)
        data = round_trip_load(self.yaml_str)
        x = data[2].items()
        print('d2 items', d[2].items(), len(d[2].items()), x, len(x))
        ref = len(d[2].items())
        print('ref', ref)
        assert len(x) == ref
        del data[2]['m']
        if PY3:
            ref -= 1
        assert len(x) == ref
        del data[2]['d']
        if PY3:
            ref -= 1
        assert len(x) == ref
        del data[2]['a']
        if PY3:
            ref -= 1
        assert len(x) == ref

    def test_issue_196_cast_of_dict(self, capsys):
        from srsly.ruamel_yaml import YAML
        yaml = YAML()
        mapping = yaml.load('        anchored: &anchor\n          a : 1\n\n        mapping:\n          <<: *anchor\n          b: 2\n        ')['mapping']
        for k in mapping:
            print('k', k)
        for k in mapping.copy():
            print('kc', k)
        print('v', list(mapping.keys()))
        print('v', list(mapping.values()))
        print('v', list(mapping.items()))
        print(len(mapping))
        print('-----')
        assert 'a' in mapping
        x = {}
        for k in mapping:
            x[k] = mapping[k]
        assert 'a' in x
        assert 'a' in mapping.keys()
        assert mapping['a'] == 1
        assert mapping.__getitem__('a') == 1
        assert 'a' in dict(mapping)
        assert 'a' in dict(mapping.items())

    def test_values_of_merged(self):
        from srsly.ruamel_yaml import YAML
        yaml = YAML()
        data = yaml.load(dedent(self.yaml_str))
        assert list(data[2].values()) == [1, 6, 'x2', 'x3', 'y4']

    def test_issue_213_copy_of_merge(self):
        from srsly.ruamel_yaml import YAML
        yaml = YAML()
        d = yaml.load('        foo: &foo\n          a: a\n        foo2:\n          <<: *foo\n          b: b\n        ')['foo2']
        assert d['a'] == 'a'
        d2 = d.copy()
        assert d2['a'] == 'a'
        print('d', d)
        del d['a']
        assert 'a' not in d
        assert 'a' in d2