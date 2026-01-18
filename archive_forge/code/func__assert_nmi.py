import time
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.v2 import shell
def _assert_nmi(self, server_id, timeout=60, poll_interval=1):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if 'trigger_crash_dump' in self.nova('instance-action-list %s ' % server_id):
            break
        time.sleep(poll_interval)
    else:
        self.fail("Trigger crash dump hasn't been executed for server %s" % server_id)