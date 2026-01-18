import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, YAML
class TestIndentFailures:

    def test_tag(self):
        round_trip('        !!python/object:__main__.Developer\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    def test_full_tag(self):
        round_trip('        !!tag:yaml.org,2002:python/object:__main__.Developer\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    def test_standard_tag(self):
        round_trip('        !!tag:yaml.org,2002:python/object:map\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    def test_Y1(self):
        round_trip('        !yyy\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    def test_Y2(self):
        round_trip('        !!yyy\n        name: Anthon\n        location: Germany\n        language: python\n        ')