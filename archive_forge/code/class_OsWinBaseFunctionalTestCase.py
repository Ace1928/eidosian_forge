import os
from oslotest import base
class OsWinBaseFunctionalTestCase(base.BaseTestCase):

    def setUp(self):
        super(OsWinBaseFunctionalTestCase, self).setUp()
        if not os.name == 'nt':
            raise self.skipException('os-win functional tests can only be run on Windows.')