from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
class VolumeTypeDefaultTest(utils.TestCase):

    def test_set(self):
        defaults.default_types.create('4c298f16-e339-4c80-b934-6cbfcb7525a0', '629632e7-99d2-4c40-9ae3-106fa3b1c9b7')
        defaults.assert_called('PUT', 'v3/default-types/629632e7-99d2-4c40-9ae3-106fa3b1c9b7', body={'default_type': {'volume_type': '4c298f16-e339-4c80-b934-6cbfcb7525a0'}})

    def test_get(self):
        defaults.default_types.list('629632e7-99d2-4c40-9ae3-106fa3b1c9b7')
        defaults.assert_called('GET', 'v3/default-types/629632e7-99d2-4c40-9ae3-106fa3b1c9b7')

    def test_get_all(self):
        defaults.default_types.list()
        defaults.assert_called('GET', 'v3/default-types')

    def test_unset(self):
        defaults.default_types.delete('629632e7-99d2-4c40-9ae3-106fa3b1c9b7')
        defaults.assert_called('DELETE', 'v3/default-types/629632e7-99d2-4c40-9ae3-106fa3b1c9b7')