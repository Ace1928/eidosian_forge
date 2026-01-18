from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_rdp_console_get(self):
    self._test_console_get('get-rdp-console %s rdp-html5', 'rdp-html5')