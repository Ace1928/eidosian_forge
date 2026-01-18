import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def _wait_for_instance_actions(self, server, expected_num_of_actions):
    start_time = time.time()
    while time.time() - start_time < 60:
        actions = self.client.instance_action.list(server)
        if len(actions) == expected_num_of_actions:
            break
        time.sleep(1)
    else:
        self.fail('The number of instance actions for server %s was not %d after 60 s' % (server.id, expected_num_of_actions))
    time.sleep(1)
    return timeutils.utcnow().isoformat()