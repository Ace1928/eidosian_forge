import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
class TestFullCharSetAnchors:

    def test_master_of_orion(self):
        yaml_str = '\n        - collection: &Backend.Civilizations.RacialPerk\n            items:\n                  - key: perk_population_growth_modifier\n        - *Backend.Civilizations.RacialPerk\n        '
        data = load(yaml_str)

    def test_roundtrip_00(self):
        yaml_str = '\n        - &dotted.words.here\n          a: 1\n          b: 2\n        - *dotted.words.here\n        '
        data = round_trip(yaml_str)

    def test_roundtrip_01(self):
        yaml_str = '\n        - &dotted.words.here[a, b]\n        - *dotted.words.here\n        '
        data = load(yaml_str)
        compare(data, yaml_str.replace('[', ' ['))