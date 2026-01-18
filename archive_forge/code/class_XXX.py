import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
class XXX(yaml.comments.CommentedMap):

    @staticmethod
    def yaml_dump(dumper, data):
        return dumper.represent_mapping(u'!xxx', data)

    @classmethod
    def yaml_load(cls, constructor, node):
        data = cls()
        yield data
        constructor.construct_mapping(node, data)