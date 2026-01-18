import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def _check_fields_present(self, clusters, detailed=False):
    expected_keys = {'name', 'binary', 'state', 'status'}
    if detailed:
        expected_keys.update(('num_hosts', 'num_down_hosts', 'last_heartbeat', 'disabled_reason', 'created_at', 'updated_at'))
    for cluster in clusters:
        self.assertEqual(expected_keys, set(cluster.to_dict()))