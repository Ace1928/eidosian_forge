from neutron_lib.api.definitions import bgpvpn
from neutron_lib.api import validators
from neutron_lib.tests.unit.api.definitions import base
def _data_for_valid_rtdt(self):
    values = [['1:1'], ['1:4294967295'], ['65535:0'], ['65535:4294967295'], ['1.1.1.1:1'], ['1.1.1.1:65535'], ['4294967295:0'], ['65536:65535'], ['4294967295:65535']]
    for value in values:
        yield value