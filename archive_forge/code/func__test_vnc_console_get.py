from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_vnc_console_get(self):
    self._test_console_get('get-vnc-console %s novnc', 'novnc')