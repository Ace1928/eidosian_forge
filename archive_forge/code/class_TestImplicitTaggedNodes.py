import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
class TestImplicitTaggedNodes:

    def test_scalar(self):
        round_trip('        - !Scalar abcdefg\n        ')

    def test_mapping(self):
        round_trip('        - !Mapping {a: 1, b: 2}\n        ')

    def test_sequence(self):
        yaml = YAML()
        yaml.brace_single_entry_mapping_in_flow_sequence = True
        yaml.mapping_value_align = True
        yaml.round_trip('\n        - !Sequence [a, {b: 1}, {c: {d: 3}}]\n        ')

    def test_sequence2(self):
        yaml = YAML()
        yaml.mapping_value_align = True
        yaml.round_trip('\n        - !Sequence [a, b: 1, c: {d: 3}]\n        ')