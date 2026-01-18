import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
class TestOrderedDict:

    def test_ordereddict(self):
        from collections import OrderedDict
        import srsly.ruamel_yaml
        assert srsly.ruamel_yaml.dump(OrderedDict()) == '!!omap []\n'