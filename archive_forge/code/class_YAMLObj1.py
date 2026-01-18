import re
import pytest  # NOQA
from .roundtrip import dedent
class YAMLObj1(object):
    yaml_tag = u'!obj:'

    @classmethod
    def from_yaml(cls, loader, suffix, node):
        import srsly.ruamel_yaml
        obj1 = Obj1(suffix)
        if isinstance(node, srsly.ruamel_yaml.MappingNode):
            obj1.add_node(loader.construct_mapping(node))
        else:
            raise NotImplementedError
        return obj1

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag + data._suffix, data.dump())