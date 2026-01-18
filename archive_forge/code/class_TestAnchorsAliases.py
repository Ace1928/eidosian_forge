import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
class TestAnchorsAliases:

    def test_anchor_id_renumber(self):
        from srsly.ruamel_yaml.serializer import Serializer
        assert Serializer.ANCHOR_TEMPLATE == 'id%03d'
        data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        ')
        compare(data, '\n        a: &id001\n          b: 1\n          c: 2\n        d: *id001\n        ')

    def test_template_matcher(self):
        """test if id matches the anchor template"""
        from srsly.ruamel_yaml.serializer import templated_id
        assert templated_id(u'id001')
        assert templated_id(u'id999')
        assert templated_id(u'id1000')
        assert templated_id(u'id0001')
        assert templated_id(u'id0000')
        assert not templated_id(u'id02')
        assert not templated_id(u'id000')
        assert not templated_id(u'x000')

    def test_anchor_assigned(self):
        from srsly.ruamel_yaml.comments import CommentedMap
        data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')
        d = data['d']
        assert isinstance(d, CommentedMap)
        assert d.yaml_anchor() is None
        e = data['e']
        assert isinstance(e, CommentedMap)
        assert e.yaml_anchor().value == 'etemplate'
        assert e.yaml_anchor().always_dump is False

    def test_anchor_id_retained(self):
        data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')
        compare(data, '\n        a: &id001\n          b: 1\n          c: 2\n        d: *id001\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')

    @pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
    def test_alias_before_anchor(self):
        from srsly.ruamel_yaml.composer import ComposerError
        with pytest.raises(ComposerError):
            data = load('\n            d: *id002\n            a: &id002\n              b: 1\n              c: 2\n            ')
            data = data

    def test_anchor_on_sequence(self):
        from srsly.ruamel_yaml.comments import CommentedSeq
        data = load('\n        nut1: &alice\n         - 1\n         - 2\n        nut2: &blake\n         - some data\n         - *alice\n        nut3:\n         - *blake\n         - *alice\n        ')
        r = data['nut1']
        assert isinstance(r, CommentedSeq)
        assert r.yaml_anchor() is not None
        assert r.yaml_anchor().value == 'alice'
    merge_yaml = dedent('\n        - &CENTER {x: 1, y: 2}\n        - &LEFT {x: 0, y: 2}\n        - &BIG {r: 10}\n        - &SMALL {r: 1}\n        # All the following maps are equal:\n        # Explicit keys\n        - x: 1\n          y: 2\n          r: 10\n          label: center/small\n        # Merge one map\n        - <<: *CENTER\n          r: 10\n          label: center/medium\n        # Merge multiple maps\n        - <<: [*CENTER, *BIG]\n          label: center/big\n        # Override\n        - <<: [*BIG, *LEFT, *SMALL]\n          x: 1\n          label: center/huge\n        ')

    def test_merge_00(self):
        data = load(self.merge_yaml)
        d = data[4]
        ok = True
        for k in d:
            for o in [5, 6, 7]:
                x = d.get(k)
                y = data[o].get(k)
                if not isinstance(x, int):
                    x = x.split('/')[0]
                    y = y.split('/')[0]
                if x != y:
                    ok = False
                    print('key', k, d.get(k), data[o].get(k))
        assert ok

    def test_merge_accessible(self):
        from srsly.ruamel_yaml.comments import CommentedMap, merge_attrib
        data = load('\n        k: &level_2 { a: 1, b2 }\n        l: &level_1 { a: 10, c: 3 }\n        m:\n          <<: *level_1\n          c: 30\n          d: 40\n        ')
        d = data['m']
        assert isinstance(d, CommentedMap)
        assert hasattr(d, merge_attrib)

    def test_merge_01(self):
        data = load(self.merge_yaml)
        compare(data, self.merge_yaml)

    def test_merge_nested(self):
        yaml = '\n        a:\n          <<: &content\n            1: plugh\n            2: plover\n          0: xyzzy\n        b:\n          <<: *content\n        '
        data = round_trip(yaml)

    def test_merge_nested_with_sequence(self):
        yaml = '\n        a:\n          <<: &content\n            <<: &y2\n              1: plugh\n            2: plover\n          0: xyzzy\n        b:\n          <<: [*content, *y2]\n        '
        data = round_trip(yaml)

    def test_add_anchor(self):
        from srsly.ruamel_yaml.comments import CommentedMap
        data = CommentedMap()
        data_a = CommentedMap()
        data['a'] = data_a
        data_a['c'] = 3
        data['b'] = 2
        data.yaml_set_anchor('klm', always_dump=True)
        data['a'].yaml_set_anchor('xyz', always_dump=True)
        compare(data, '\n        &klm\n        a: &xyz\n          c: 3\n        b: 2\n        ')

    def test_reused_anchor(self):
        from srsly.ruamel_yaml.error import ReusedAnchorWarning
        yaml = '\n        - &a\n          x: 1\n        - <<: *a\n        - &a\n          x: 2\n        - <<: *a\n        '
        with pytest.warns(ReusedAnchorWarning):
            data = round_trip(yaml)

    def test_issue_130(self):
        ys = dedent('        components:\n          server: &server_component\n            type: spark.server:ServerComponent\n            host: 0.0.0.0\n            port: 8000\n          shell: &shell_component\n            type: spark.shell:ShellComponent\n\n        services:\n          server: &server_service\n            <<: *server_component\n          shell: &shell_service\n            <<: *shell_component\n            components:\n              server: {<<: *server_service}\n        ')
        data = srsly.ruamel_yaml.safe_load(ys)
        assert data['services']['shell']['components']['server']['port'] == 8000

    def test_issue_130a(self):
        ys = dedent('        components:\n          server: &server_component\n            type: spark.server:ServerComponent\n            host: 0.0.0.0\n            port: 8000\n          shell: &shell_component\n            type: spark.shell:ShellComponent\n\n        services:\n          server: &server_service\n            <<: *server_component\n            port: 4000\n          shell: &shell_service\n            <<: *shell_component\n            components:\n              server: {<<: *server_service}\n        ')
        data = srsly.ruamel_yaml.safe_load(ys)
        assert data['services']['shell']['components']['server']['port'] == 4000